"""CV sull'intero dataset: ogni gruppo è test una volta; validation interna separata."""
from pathlib import Path
import csv
import json
import warnings
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from training.config import digest, plain, write_json
from training.data import index_dataset, split_validation, fingerprint, save_index


def make_folds(records, n_splits=5, seed=42, validation_fraction=.2):
    # Le vecchie cartelle training/validation/test vengono riunite, senza spostare i WAV.
    # Un eventuale test_finale esplicito non entra nella CV.
    pool = [dict(r) for r in records if r['subset'] != 'test_finale']
    for r in pool:
        r['source_subset'] = r['subset']
        r['subset'] = 'pool'
    if n_splits < 2:
        raise ValueError('Servono almeno due fold')
    people = [i for i,r in enumerate(pool) if r['group'].startswith('P')]
    environments = sorted({r['group'] for r in pool if r['group'].startswith('A')})
    if len(environments) < n_splits:
        raise ValueError(f'{n_splits} fold richiedono almeno {n_splits} ambienti noise indipendenti; '
                         f'ne risultano {len(environments)}. Non si dividono gli ambienti per creare fold.')
    if len({pool[i]['group'] for i in people}) < n_splits:
        raise ValueError('Persone insufficienti per i fold richiesti')
    # Gruppo e label sono indipendenti: un ritaglio Axx può essere unknown
    # dopo la revisione in Edge Impulse. Rimane con il suo ambiente, senza relabel.
    original_labels = ('eva','noise','unknown')
    total = np.array([sum(r['original_label']==l for r in pool) for l in original_labels],float)
    y = np.array([pool[i]['original_label'] for i in people])
    groups = np.array([pool[i]['group'] for i in people])
    # Assegna ambienti interi: con cinque ambienti/cinque fold, esattamente uno per fold.
    env_counts={g:sum(r['group']==g for r in pool) for g in environments}
    env_assignment={}; loads=np.zeros(n_splits)
    for g in sorted(environments,key=lambda g:(-env_counts[g],g)):
        f=int(np.argmin(loads));env_assignment[g]=f;loads[f]+=env_counts[g]
    best=None
    for attempt in range(100):
        splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed+attempt)
        assignment=dict(env_assignment)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            for f,(_,indices) in enumerate(splitter.split(np.zeros(len(y)),y,groups)):
                for g in set(groups[indices]):assignment[g]=f
        counts=np.array([[sum(assignment[r['group']]==f and r['original_label']==l for r in pool)
                          for l in original_labels] for f in range(n_splits)],float)
        if np.any(counts==0):continue
        # Selezione solo sulla distribuzione, mai sulle prestazioni del modello.
        score=float(np.mean((counts/total[None,:]-1/n_splits)**2))
        if best is None or score<best[0]:best=(score,assignment)
    if best is None:
        raise ValueError('Impossibile creare fold con tutte le classi e gruppi separati')
    assignment=best[1]
    folds=[]
    for f in range(n_splits):
        remaining=[dict(r,subset='train') for r in pool if assignment[r['group']]!=f]
        # Non utilizza il test esterno per early stopping o normalizzazione.
        inside=split_validation(remaining,validation_fraction,seed+1000+f)
        mapping={r['relative']:r['subset'] for r in inside}
        fold=[dict(r,subset=('test' if assignment[r['group']]==f else mapping[r['relative']])) for r in pool]
        folds.append(fold)
    return folds,assignment


def save_fold_assignment(records, folds, assignment, folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'group_folds.json',{g:f+1 for g,f in assignment.items()})
    rows=[]
    for fold_number,fold in enumerate(folds,1):
        save_index(fold,folder/f'fold_{fold_number:02}_manifest.csv')
        row={'fold':fold_number}
        for subset in ['train','validation','test']:
            selected=[r for r in fold if r['subset']==subset]
            row[subset+'_samples']=len(selected)
            row[subset+'_fraction']=len(selected)/len(fold)
            row[subset+'_groups']=' '.join(sorted({r['group'] for r in selected}))
            for label in ['eva','noise','unknown']:
                row[f'{subset}_{label}']=sum(r['original_label']==label for r in selected)
        rows.append(row)
    with (folder/'fold_sizes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)


def run_cross_validation(settings, feature_options, model_options, display_plots=False, save_plots=None):
    from copy import deepcopy
    from training.experiments import run_one, write_table
    from training.cv_reports import summarize_configuration, flatten_report
    if save_plots is None:save_plots=display_plots
    records=index_dataset(settings.data_dir,settings.binary)
    folds,assignment=make_folds(records,settings.n_folds,settings.split_seed,settings.validation_fraction)
    base=Path(settings.output_dir)/'cross_validation'
    base.mkdir(parents=True,exist_ok=True)
    dataset_id=digest({'pool':fingerprint(records),'groups':assignment,'n_folds':settings.n_folds,
                       'validation_fraction':settings.validation_fraction,'split_seed':settings.split_seed})[:16]
    destination=base/dataset_id
    save_fold_assignment(records,folds,assignment,destination/'splits')
    all_rows=[]; summary_rows=[]
    count=len(feature_options)*len(model_options)*len(settings.seeds)*len(folds)
    print(f'CV: {settings.n_folds} fold, {count} addestramenti. Vecchi train/test riuniti in memoria.')
    print('Dimensioni e gruppi:',destination/'splits'/'fold_sizes.csv')
    print('Ogni persona/ambiente è test una volta per seed e configurazione.')
    for feat in feature_options:
        for model_cfg in model_options:
            training_identity={k:v for k,v in plain(settings).items() if k not in
                               ('data_dir','output_dir','cache_dir','save_plots','show_plots',
                                'cross_validation','n_folds','split_seed','validation_fraction')}
            config_id=digest({'features':plain(feat),'model':plain(model_cfg),
                              'training':training_identity})[:12]
            config_folder=destination/f'{model_cfg.family}_{config_id}'
            runs=[]
            for seed in settings.seeds:
                for number,fold in enumerate(folds,1):
                    local=deepcopy(settings)
                    local.output_dir=config_folder/f'seed_{seed}'/f'fold_{number:02}'
                    local.show_plots=display_plots
                    # In grid non si producono PNG; metriche/history/OOF restano salvate.
                    local.save_plots=save_plots
                    print(f'\n{model_cfg.family} / seed {seed} / fold {number}/{len(folds)}',flush=True)
                    row=run_one(local,fold,feat,model_cfg,seed)
                    row=dict(row,fold=number,cv_seed=seed,
                             run_folder=str(local.output_dir/row['run_id']),cv_configuration=config_id)
                    for prefix in ['validation','test']:
                        report=json.loads((Path(row['run_folder'])/f'{prefix}_metrics.json').read_text())
                        row.update(flatten_report(report,prefix))
                    runs.append(row);all_rows.append(row)
                    write_table(destination/'cv_fold_metrics.csv',all_rows)
            result=summarize_configuration(runs,config_folder,settings.n_folds,
                                           show=display_plots,save_plots=save_plots)
            result['summary_folder'] = str(config_folder)
            summary_rows.append(result)
            summary_rows.sort(key=lambda r:r['test_eva_f1_mean'],reverse=True)
            write_table(destination/'cv_architecture_summary.csv',summary_rows)
    print('CV completata:',destination/'cv_architecture_summary.csv')
    return summary_rows
