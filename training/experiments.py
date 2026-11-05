"""Training riproducibile e cache. Test di sviluppo sempre attivo dopo ciascun esperimento."""
from pathlib import Path
from itertools import product
from datetime import datetime
import csv
import json
import time
import platform
import numpy as np
import tensorflow as tf
from training.config import plain, digest, write_json, MFCCConfig, ModelConfig
from training.data import index_dataset, split_validation, fingerprint, save_index, describe, training_class_weights
from training.features import feature_matrix
from training.models import build_model
from training.metrics import save_evaluation, plot_history
from training.keras_metrics import EvaAUC


def make_dataset(x, y, settings, training, seed, mean=None, variance=None):
    ds = tf.data.Dataset.from_tensor_slices((x,y))
    options = tf.data.Options()
    options.experimental_deterministic = settings.deterministic
    options.threading.private_threadpool_size = 1
    ds = ds.with_options(options)
    if training:
        ds = ds.shuffle(settings.batch_size*4,seed=seed,reshuffle_each_iteration=True)
        width_max = min(settings.time_mask_max_frames, x.shape[1])
        if width_max > 0:
            # mask_with_mean=False: zeri sulle feature DSP, senza normalizzazione aggiuntiva.
            center = tf.zeros((x.shape[-1],), dtype=tf.float32)
            def mask(a,b):
                width = tf.random.uniform((),0,width_max+1,dtype=tf.int32)
                start = tf.random.uniform((),0,tf.shape(a)[0]-width+1,dtype=tf.int32)
                t = tf.range(tf.shape(a)[0])
                keep = tf.logical_or(t < start,t >= start+width)
                return tf.where(keep[:,None],a,center[None,:]), b
            ds = ds.map(mask,num_parallel_calls=1)
    return ds.batch(settings.batch_size).prefetch(1)


def prepare(settings):
    records = split_validation(index_dataset(settings.data_dir,settings.binary),
                               settings.validation_fraction,settings.split_seed)
    settings.output_dir = Path(settings.output_dir); settings.output_dir.mkdir(parents=True,exist_ok=True)
    snapshot = settings.output_dir/'datasets'/fingerprint(records)
    snapshot.mkdir(parents=True,exist_ok=True)
    save_index(records,snapshot/'manifest.csv')
    describe(records)
    selected = [r for r in records if r['subset'] == 'test']
    labels = {'eva', 'non_target'} if settings.binary else {'eva', 'noise', 'unknown'}
    if {r['label'] for r in selected} != labels:
        raise ValueError('Per il test automatico servono tutte le classi in test/test_sviluppo. '
                         'Non viene usato test_finale.')
    return records


def run_one(settings, records, feature_cfg, model_cfg, seed):
    labels = ('eva','non_target') if settings.binary else ('eva','noise','unknown')
    signature = dict(settings=plain(settings),mfcc=plain(feature_cfg),model=plain(model_cfg),
                     seed=seed,dataset_fingerprint=fingerprint(records),
                     tensorflow=tf.__version__,numpy=np.__version__,pipeline_version=5)
    identity=dict(signature,settings={k:v for k,v in signature['settings'].items()
                                     if k not in ('save_plots','show_plots')})
    run_id = f'{model_cfg.family}_mfcc{feature_cfg.n_mfcc}_mel{feature_cfg.n_mels}_{digest(identity)[:12]}'
    folder = settings.output_dir/run_id
    if (folder/'done.json').exists():
        print('Già completato, riuso:',run_id)
        if settings.save_plots:
            from training.cv_reports import plot_fold
            plot_fold(folder,show=settings.show_plots)
        return json.loads((folder/'done.json').read_text())
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'config.json',signature)
    save_index(records,folder/'manifest.csv')
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if settings.deterministic:
        tf.config.experimental.enable_op_determinism()
    train = [r for r in records if r['subset']=='train']
    val = [r for r in records if r['subset']=='validation']
    xtrain = feature_matrix(train,feature_cfg,settings.cache_dir,settings.length_tolerance_ms)
    xval = feature_matrix(val,feature_cfg,settings.cache_dir,settings.length_tolerance_ms)
    ytrain = np.array([labels.index(r['label']) for r in train],dtype=np.int32)
    yval = np.array([labels.index(r['label']) for r in val],dtype=np.int32)
    weights,train_counts=training_class_weights(ytrain,len(labels),settings.class_weight_mode)
    write_json(folder/'training_class_weights.json',dict(mode=settings.class_weight_mode,
               counts=dict(zip(labels,train_counts)),weights=None if weights is None else
               {labels[i]:w for i,w in weights.items()}))
    mean = variance = None
    write_json(folder/'preprocessing.json',dict(mfcc=plain(feature_cfg),extra_normalization=False,
               temporal_mask_fill=0.0,temporal_mask_width_range=[0,settings.time_mask_max_frames]))
    model = build_model(xtrain.shape[1:],len(labels),model_cfg,mean,variance,settings.gaussian_std)
    model.compile(optimizer=tf.keras.optimizers.Adam(settings.learning_rate),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy',EvaAUC()])
    with (folder/'model_summary.txt').open('w') as f:
        model.summary(print_fn=lambda s:f.write(s+'\n'))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(str(folder/'best.keras'),monitor='val_loss',save_best_only=True),
                 tf.keras.callbacks.CSVLogger(str(folder/'history.csv')),
                 tf.keras.callbacks.TerminateOnNaN()]
    if settings.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=settings.patience,
                                                          restore_best_weights=True))
    print('\nTRAINING',run_id, 'shape',xtrain.shape,'classi',labels,flush=True)
    begin = time.perf_counter()
    history = model.fit(make_dataset(xtrain,ytrain,settings,True,seed,mean,variance),
                        validation_data=make_dataset(xval,yval,settings,False,seed),
                        epochs=settings.epochs,callbacks=callbacks,class_weight=weights,verbose=2).history
    elapsed = time.perf_counter()-begin
    if not (folder/'best.keras').exists():
        raise RuntimeError('Nessun checkpoint valido; verificare NaN/loss')
    best = tf.keras.models.load_model(folder/'best.keras',compile=False)
    scores = best.predict(make_dataset(xval,yval,settings,False,seed),verbose=0)
    report = save_evaluation(folder,'validation',val,yval,scores,labels)
    selected = [r for r in records if r['subset'] == 'test']
    test_report = evaluate_records(best, selected, feature_cfg, settings, seed, labels, folder, 'test')
    print('TEST DI SVILUPPO: accuracy={:.3f}, EVA precision={:.3f}, recall={:.3f}, F1={:.3f}'.format(
        test_report['accuracy'], test_report['per_class']['eva']['precision'],
        test_report['per_class']['eva']['recall'], test_report['per_class']['eva']['f1']))
    if settings.save_plots:
        from training.cv_reports import plot_fold
        plot_fold(folder, show=settings.show_plots)

    eva = report['per_class']['eva']
    summary = dict(run_id=run_id,configuration_id=digest(dict(mfcc=plain(feature_cfg),model=plain(model_cfg))),
                   seed=seed,batch_size=settings.batch_size,learning_rate=settings.learning_rate,
                   class_weight_mode=settings.class_weight_mode,family=model_cfg.family,filters='-'.join(map(str,model_cfg.filters)),
                   kernel=model_cfg.kernel,dropout=model_cfg.dropout,head=model_cfg.head,
                   n_mfcc=feature_cfg.n_mfcc,n_mels=feature_cfg.n_mels,hop_ms=feature_cfg.hop_ms,
                   val_accuracy=report['accuracy'],val_loss=report['loss'],val_eva_auc=eva['auc_roc'],
                   eva_precision=eva['precision'],eva_recall=eva['recall'],eva_f1=eva['f1'],
                   eva_fpr=eva['false_positive_rate'],parameters=best.count_params(),
                   best_epoch=int(np.argmin(history['val_loss'])+1),epochs_run=len(history['loss']),
                   training_seconds=elapsed,model_bytes=(folder/'best.keras').stat().st_size,
                   # .keras include metadati/optimizer: NON memoria microcontrollore.
                   dataset_fingerprint=fingerprint(records))
    test_eva = test_report['per_class']['eva']
    summary.update(test_accuracy=test_report['accuracy'], test_loss=test_report['loss'],test_eva_auc=test_eva['auc_roc'],
                   test_eva_precision=test_eva['precision'], test_eva_recall=test_eva['recall'],
                   test_eva_f1=test_eva['f1'], test_eva_fpr=test_eva['false_positive_rate'])
    write_json(folder/'environment.json',dict(python=platform.python_version(),tensorflow=tf.__version__,
                                            devices=[str(d) for d in tf.config.list_physical_devices()]))
    write_json(folder/'done.json',summary)
    return summary


def run_search(settings, feature_options, model_options):
    records = prepare(settings)
    rows = []
    combos = list(product(feature_options,model_options,settings.seeds))
    print(f'{len(combos)} addestramenti richiesti; quelli completati vengono riusati.')
    for i,(feat,model,seed) in enumerate(combos,1):
        print(f'\nEsperimento {i}/{len(combos)}',flush=True)
        rows.append(run_one(settings,records,feat,model,seed))
        # Risultati di QUESTA ricerca, senza mescolare dataset diversi.
        write_table(settings.output_dir/'search_runs.csv', rows)
        aggregate(rows,settings.output_dir/'search_summary.csv')
    print('Completato. Ordinamento indicativo per F1 EVA; verificare anche recall/FPR e costo.')
    return rows


def write_table(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def aggregate(rows,path):
    result=[]
    for key in sorted({r['configuration_id'] for r in rows}):
        group=[r for r in rows if r['configuration_id']==key]
        row={k:group[0][k] for k in ['configuration_id','family','filters','kernel','dropout','head','n_mfcc','n_mels','hop_ms','parameters']}
        row['repetitions']=len(group)
        metrics = ['eva_precision','eva_recall','eva_f1','eva_fpr','val_accuracy']
        if all('test_accuracy' in r for r in group):
            metrics += ['test_accuracy','test_eva_precision','test_eva_recall','test_eva_f1','test_eva_fpr']
        for metric in metrics:
            values=[r[metric] for r in group]
            row[metric+'_mean']=float(np.mean(values));row[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.0
        result.append(row)
    result.sort(key=lambda r:r['eva_f1_mean'],reverse=True)
    write_table(path,result)


def evaluate_saved(run_folder, subset='test'):
    """Valutazione manuale aggiuntiva, inclusa quella finale indipendente."""
    if subset not in ('test','test_finale'):
        raise ValueError('Usare test o test_finale')
    folder=Path(run_folder)
    saved=json.loads((folder/'config.json').read_text())
    data_root=Path(saved['settings']['data_dir'])
    from training.config import Settings
    s=Settings()
    for k,v in saved['settings'].items():setattr(s,k,v)
    if s.cross_validation:
        from training.cross_validation import make_folds
        candidates,_=make_folds(index_dataset(data_root,s.binary),s.n_folds,s.split_seed,s.validation_fraction)
        matches=[r for r in candidates if fingerprint(r)==saved['dataset_fingerprint']]
        if len(matches)!=1:raise ValueError('Impossibile ricostruire il fold del modello')
        all_records=matches[0]
    else:
        all_records=split_validation(index_dataset(data_root,s.binary),s.validation_fraction,s.split_seed)
    if fingerprint(all_records)!=saved['dataset_fingerprint']:
        raise ValueError('Dataset cambiato rispetto al training. Non confrontare versioni diverse: creare una nuova ricerca.')
    selected=[r for r in all_records if r['subset']==subset]
    if not selected:raise ValueError(f'Nessun file in {subset}')
    labels=('eva','non_target') if s.binary else ('eva','noise','unknown')
    if {r['label'] for r in selected}!=set(labels):
        raise ValueError('Il test deve contenere tutte le classi')
    feat=MFCCConfig(**saved['mfcc'])
    model=tf.keras.models.load_model(folder/'best.keras',compile=False)
    return evaluate_records(model, selected, feat, s, saved['seed'], labels, folder, subset)


def evaluate_records(model, records, cfg, settings, seed, labels, folder, prefix):
    x = feature_matrix(records, cfg, settings.cache_dir, settings.length_tolerance_ms)
    y = np.array([labels.index(r['label']) for r in records], dtype=np.int32)
    scores = model.predict(make_dataset(x, y, settings, False, seed), verbose=0)
    return save_evaluation(folder, prefix, records, y, scores, labels)
