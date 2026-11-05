# %% CONFIGURAZIONE — aprire questo file in Spyder ed eseguire con F5
"""Un training finale binario + esportazione float32 e full-int8.

Le metriche di validation sono controlli di sviluppo, non un nuovo test indipendente.
Gli audio storici train/test sono riuniti in memoria e risuddivisi per gruppo;
la cartella test_finale, se presente, rimane esclusa.
"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training.config import Settings, MFCCConfig, ModelConfig
from deployment.tflite_export import export_models

SETTINGS = Settings(
    data_dir=PROJECT/'data_v2', output_dir=PROJECT/'results'/'final_model',
    cache_dir=PROJECT/'cache', binary=True, validation_fraction=.20,
    split_seed=42, seeds=(42,), epochs=100, batch_size=128,
    learning_rate=.005, early_stopping=False, gaussian_std=.45,
    time_mask_max_frames=1, class_weight_mode='none',
    show_plots=False, save_plots=True,
)
FEATURES = MFCCConfig(n_mfcc=13, n_mels=40)
NETWORK = ModelConfig(family='cnn1d', filters=(8,16), kernel=5, dropout=.25)
CALIBRATION_SAMPLES = 200  # Solo feature originali del training, senza augmentation.
# Dopo un'interruzione dell'export, impostare la cartella già creata per non riaddestrare.
# Esempio: PROJECT/'results'/'final_model'/'20261105_120000_123456'
REUSE_RUN = None

# %% IMPLEMENTAZIONE

def final_split(settings):
    from training.data import index_dataset, split_validation, _check_overlap
    records = index_dataset(settings.data_dir, binary=True)
    # Ogni gruppo può comparire in un solo subset. Non si spostano i WAV.
    excluded = [dict(r) for r in records if r['subset']=='test_finale']
    pool = [dict(r, subset='train') for r in records if r['subset']!='test_finale']
    split = split_validation(pool, settings.validation_fraction, settings.split_seed)
    _check_overlap(split + excluded)
    return split, excluded


def main():
    import json
    from datetime import datetime
    import numpy as np
    import tensorflow as tf
    from training.config import plain, write_json
    from training.data import fingerprint, save_index, describe
    from training.features import feature_matrix
    from training.experiments import make_dataset
    from training.models import build_model
    from training.keras_metrics import EvaAUC
    from training.metrics import plot_history
    if not SETTINGS.binary or len(SETTINGS.seeds)!=1:
        raise ValueError('Questo script richiede binary=True e un solo seed')
    seed = SETTINGS.seeds[0]
    records, excluded = final_split(SETTINGS)
    describe(records)
    signature = dict(settings=plain(SETTINGS), mfcc=plain(FEATURES), model=plain(NETWORK),
                     dataset_fingerprint=fingerprint(records), seed=seed,
                     tensorflow=tf.__version__, pipeline='final_binary_v1')
    folder = Path(REUSE_RUN) if REUSE_RUN else SETTINGS.output_dir/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if REUSE_RUN:
        saved = json.loads((folder/'config.json').read_text())
        if any(saved[k]!=signature[k] for k in signature):
            raise ValueError('Configurazione/dataset/versione diversi dal training da riusare')
    else:
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder/'config.json', signature)
        save_index(records, folder/'manifest.csv')
        if excluded:
            save_index(excluded, folder/'excluded_test_finale.csv')
    print('Cartella finale:',folder,flush=True)
    train = [r for r in records if r['subset']=='train']
    val = [r for r in records if r['subset']=='validation']
    xtrain = feature_matrix(train, FEATURES, SETTINGS.cache_dir, SETTINGS.length_tolerance_ms)
    xval = feature_matrix(val, FEATURES, SETTINGS.cache_dir, SETTINGS.length_tolerance_ms)
    labels = ('eva','non_target')
    ytrain = np.array([labels.index(r['label']) for r in train], dtype=np.int32)
    yval = np.array([labels.index(r['label']) for r in val], dtype=np.int32)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if SETTINGS.deterministic:
        tf.config.experimental.enable_op_determinism()
    if not REUSE_RUN:
        model = build_model(xtrain.shape[1:], 2, NETWORK, None, None, SETTINGS.gaussian_std)
        model.compile(optimizer=tf.keras.optimizers.Adam(SETTINGS.learning_rate),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy',EvaAUC()])
        history = model.fit(make_dataset(xtrain,ytrain,SETTINGS,True,seed),
            validation_data=make_dataset(xval,yval,SETTINGS,False,seed), epochs=SETTINGS.epochs,
            callbacks=[tf.keras.callbacks.ModelCheckpoint(str(folder/'best.keras'),monitor='val_loss',save_best_only=True),
                       tf.keras.callbacks.CSVLogger(str(folder/'history.csv')),
                       tf.keras.callbacks.TerminateOnNaN()], verbose=2).history
        if not np.isfinite(history['val_loss']).all():
            raise RuntimeError('Validation loss non finita: export interrotto')
        plot_history(history, folder/'training_curves.png')
        write_json(folder/'training_summary.json',dict(best_epoch=int(np.argmin(history['val_loss'])+1),
            epochs_run=len(history['loss']), parameters=model.count_params(),
            train_samples=len(train), validation_samples=len(val)))
    best = tf.keras.models.load_model(folder/'best.keras', compile=False)
    export_models(best,xtrain,ytrain,xval,yval,val,folder,FEATURES,NETWORK,seed,CALIBRATION_SAMPLES)
    write_json(folder/'complete.json',dict(status='training_and_exports_complete',
               warning='Validation checks only; deployment not yet validated on ESP32.'))
    print('\nCOMPLETATO:',folder,'\nModelli: best.keras, eva_float32.tflite, eva_int8.tflite',flush=True)
    return folder

if __name__=='__main__':
    main()
