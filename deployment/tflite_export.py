"""Float32/full-int8 conversion and inference checks; no training on import."""
from pathlib import Path

def calibration_indices(y, maximum, seed):
    """Campione riproducibile bilanciato per classe, esclusivamente dal training."""
    import numpy as np
    classes = np.unique(y)
    if maximum < len(classes):
        raise ValueError('Servono almeno tanti campioni di calibrazione quante classi')
    rng = np.random.default_rng(seed)
    pools = [rng.permutation(np.flatnonzero(y==c)).tolist() for c in classes]
    selected = []
    while len(selected) < min(maximum, len(y)) and any(pools):
        for pool in pools:
            if pool and len(selected) < min(maximum, len(y)):
                selected.append(pool.pop())
    return np.asarray(selected, dtype=int)


def lite_predictions(path, x):
    import numpy as np
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    if tuple(inp['shape'][1:]) != tuple(x.shape[1:]):
        raise ValueError('Shape TFLite incompatibile con le MFCC')
    predictions = []
    for sample in x:
        value = sample[None].astype(np.float32)
        if np.issubdtype(inp['dtype'], np.integer):
            scale, zero = inp['quantization']
            if scale <= 0:
                raise ValueError('Scala di quantizzazione input non valida')
            limits = np.iinfo(inp['dtype'])
            value = np.clip(np.rint(value/scale + zero), limits.min, limits.max)
        interpreter.set_tensor(inp['index'], value.astype(inp['dtype']))
        interpreter.invoke()
        value = interpreter.get_tensor(out['index']).astype(np.float32)
        if np.issubdtype(out['dtype'], np.integer):
            scale, zero = out['quantization']
            if scale <= 0:
                raise ValueError('Scala di quantizzazione output non valida')
            value = (value-zero)*scale
        predictions.append(value[0])
    def describe(detail):
        scale, zero = detail['quantization']
        return dict(shape=detail['shape'].tolist(), dtype=detail['dtype'].__name__,
                    scale=float(scale), zero_point=int(zero))
    return np.asarray(predictions), dict(input=describe(inp), output=describe(out))


def export_models(best, xtrain, ytrain, xval, yval, val_records, folder,
                  features, network, seed=42, calibration_samples=200):
    import numpy as np
    import tensorflow as tf
    from training.models import build_model
    from training.config import plain, write_json
    from training.metrics import save_evaluation
    # Ricostruisce la rete senza GaussianNoise e con dropout=0 (solo inferenza).
    from dataclasses import replace
    inference = build_model(xtrain.shape[1:], 2, replace(network, dropout=0.),
                            None, None, noise_std=0.)
    inference.set_weights(best.get_weights())
    original = best.predict(xval, batch_size=128, verbose=0)
    reference = inference.predict(xval, batch_size=128, verbose=0)
    np.testing.assert_allclose(original, reference, rtol=1e-5, atol=1e-6)
    inference.save(folder/'eva_inference.keras')
    # Concrete signature fissa: un segmento MFCC per invocazione.
    saved = folder/'saved_model'
    signature = [tf.TensorSpec((1, *xtrain.shape[1:]), tf.float32, name='mfcc')]
    inference.export(str(saved), input_signature=signature)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    float_path = folder/'eva_float32.tflite'
    float_path.write_bytes(converter.convert())
    indices = calibration_indices(ytrain, calibration_samples, seed)
    np.save(folder/'calibration_train_indices.npy', indices)
    def representative():
        for i in indices:
            yield [xtrain[i:i+1].astype(np.float32)]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    int_path = folder/'eva_int8.tflite'
    int_path.write_bytes(converter.convert())
    report = {}
    labels = ('eva', 'non_target')
    save_evaluation(folder, 'validation_keras', val_records, yval, reference, labels)
    for name, path in [('float32',float_path), ('int8',int_path)]:
        scores, io = lite_predictions(path, xval)
        if not np.isfinite(scores).all():
            raise ValueError(f'Predizioni {name} non finite')
        if name=='int8' and (io['input']['dtype']!='int8' or io['output']['dtype']!='int8'):
            raise ValueError('Export non interamente int8 agli ingressi/uscite')
        metrics = save_evaluation(folder, 'validation_'+name, val_records, yval, scores, labels)
        report[name] = dict(bytes=path.stat().st_size, io=io, metrics=metrics,
            max_absolute_score_difference=float(np.max(np.abs(scores-reference))),
            argmax_agreement_with_keras=float(np.mean(scores.argmax(1)==reference.argmax(1))))
        print(name, 'byte:',path.stat().st_size,'accuracy:',metrics['accuracy'],
              'EVA:',metrics['per_class']['eva'],flush=True)
    write_json(folder/'export_comparison.json', report)
    write_json(folder/'deployment.json',dict(labels=labels, eva_index=0,
        mfcc=plain(features), input_shape=[1,*xtrain.shape[1:]],
        audio_input='PCM16 values cast to float, NOT divided by 32768',
        preprocessing_in_tflite=False, calibration_source='training only; no augmentation',
        calibration_samples=len(indices), quantization_io=report['int8']['io'],
        note='MFCC C++ parity and continuous/hardware tests still required.'))
    return report


