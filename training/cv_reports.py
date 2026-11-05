"""Grafici per fold e medie CV; nessuna finestra/PNG nella grid search."""
from pathlib import Path
import csv
import json
import numpy as np
from training.config import write_json
from training.metrics import classification_metrics


def load_history(folder):
    with (Path(folder)/'history.csv').open() as f:
        rows=list(csv.DictReader(f))
    return {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}


def normalized_confusion(matrix):
    a=np.asarray(matrix,float)
    return np.divide(a,a.sum(axis=1,keepdims=True),out=np.zeros_like(a),where=a.sum(axis=1,keepdims=True)!=0)


def finish(fig,path,show):
    from matplotlib import pyplot as plt
    fig.tight_layout();fig.savefig(path,dpi=160)
    if show:
        plt.show(block=False)
    else:
        plt.close(fig)


def confusion_figure(matrix,labels,title,std=None,normalized=False):
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    a=np.asarray(matrix)
    image=ax.imshow(a,cmap='Blues',vmin=0,vmax=1 if normalized else None)
    fig.colorbar(image,ax=ax,label='Frazione della classe vera' if normalized else 'Campioni')
    for i in range(len(labels)):
        for j in range(len(labels)):
            value=f'{a[i,j]:.1%}' if normalized else str(int(a[i,j]))
            if std is not None:value+=f'\n±{std[i,j]:.1%}'
            ax.text(j,i,value,ha='center',va='center',color='white' if a[i,j]>.6*max(a.max(),1e-9) else 'black')
    ax.set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,
           yticklabels=labels,xlabel='Classe predetta',ylabel='Classe vera',title=title)
    return fig


def plot_fold(folder,show=False):
    from matplotlib import pyplot as plt
    folder=Path(folder);h=load_history(folder);epochs=np.arange(1,len(h['loss'])+1)
    best=int(np.argmin(h['val_loss'])+1)
    test_path=folder/'test_metrics.json'
    test=json.loads(test_path.read_text()) if test_path.exists() else None
    for metric,title in [('eva_auc','AUC ROC EVA vs resto'),('loss','Loss')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for key,label in [(metric,'Training (con augmentation)'),('val_'+metric,'Validation')]:
            ax.plot(epochs,h[key],label=label)
        ax.axvline(best,color='gray',linestyle=':',label=f'Checkpoint epoca {best}')
        if test:
            value=test['per_class']['eva']['auc_roc'] if metric=='eva_auc' else test['loss']
            ax.axhline(value,color='black',linestyle='--',label='Test: solo checkpoint scelto')
        ax.set(xlabel='Epoca',ylabel=title,title=f'{title} — {folder.parent.name}')
        if metric=='eva_auc':ax.set_ylim(0,1.02)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
        finish(fig,folder/f'{metric}_epochs.png',show)
    if test:
        fig=confusion_figure(test['confusion_matrix'],test['labels'],'Confusion matrix — test del fold')
        finish(fig,folder/'confusion_matrix.png',show)


def curve_statistics(histories):
    common=min(len(h['loss']) for h in histories)
    rows=[]
    for i in range(common):
        row={'epoch':i+1,'contributing_runs':len(histories)}
        for metric in ['loss','val_loss','eva_auc','val_eva_auc']:
            values=[h[metric][i] for h in histories]
            row[metric+'_mean']=float(np.mean(values))
            row[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.0
        rows.append(row)
    return rows


def flatten_report(report,prefix):
    row={f'{prefix}_accuracy':report['accuracy'],f'{prefix}_loss':report['loss']}
    row.update({f'{prefix}_{k}':v for k,v in report.get('averages',{}).items()})
    for label,values in report['per_class'].items():
        for metric,value in values.items():row[f'{prefix}_{label}_{metric}']=value
    for i,label in enumerate(report['labels']):
        for j,predicted in enumerate(report['labels']):
            row[f'{prefix}_cm_true_{label}_pred_{predicted}']=report['confusion_matrix'][i][j]
    return row


def summarize_configuration(runs,folder,n_folds,show=False,save_plots=True):
    from training.experiments import write_table
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    test_reports=[json.loads((Path(r['run_folder'])/'test_metrics.json').read_text()) for r in runs]
    labels=test_reports[0]['labels']
    summary={k:runs[0][k] for k in ['cv_configuration','family','filters','kernel','dropout','head','n_mfcc','n_mels','hop_ms','parameters','batch_size','learning_rate','class_weight_mode']}
    summary.update(n_folds=n_folds,seeds=len({r['cv_seed'] for r in runs}),evaluations=len(runs))
    flattened=[flatten_report(report,'test') for report in test_reports]
    for name in flattened[0]:
        values=[r[name] for r in flattened]
        if any(v is None for v in values):
            summary[name+'_mean']=None;summary[name+'_std']=None
        else:
            summary[name+'_mean']=float(np.mean(values))
            summary[name+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else 0.0
    normalized=np.stack([normalized_confusion(r['confusion_matrix']) for r in test_reports])
    raw=np.stack([np.asarray(r['confusion_matrix']) for r in test_reports])
    mean=normalized.mean(axis=0);std=normalized.std(axis=0,ddof=1) if len(runs)>1 else np.zeros_like(mean)
    write_json(folder/'confusion_summary.json',dict(labels=labels,
               normalized_fold_mean=mean.tolist(),normalized_fold_std=std.tolist(),
               raw_fold_mean=raw.mean(axis=0).tolist(),raw_sum=raw.sum(axis=0).tolist()))
    histories=[load_history(r['run_folder']) for r in runs]
    curves=curve_statistics(histories)
    write_table(folder/'mean_epoch_metrics.csv',curves)
    # Predizioni OOF: ogni WAV compare una sola volta per seed, sempre da modello non addestrato su quel gruppo.
    all_predictions=[]
    for run in runs:
        with (Path(run['run_folder'])/'test_predictions.csv').open() as f:
            for row in csv.DictReader(f):all_predictions.append(dict(seed=run['cv_seed'],fold=run['fold'],**row))
    write_table(folder/'out_of_fold_predictions.csv',all_predictions)
    for seed in sorted({r['cv_seed'] for r in runs}):
        selected=[r for r in all_predictions if r['seed']==seed]
        names=[r['file'] for r in selected]
        if len(names)!=len(set(names)):raise ValueError('Un file compare due volte nel test dello stesso seed')
        y=np.array([labels.index(r['true_label']) for r in selected])
        scores=np.array([[float(r['score_'+l]) for l in labels] for r in selected])
        write_json(folder/f'oof_metrics_seed_{seed}.json',classification_metrics(y,scores,labels))
    write_json(folder/'cv_summary.json',summary)
    # Separare variabilità tra fold da variabilità tra inizializzazioni.
    seed_rows=[]
    for seed in sorted({r['cv_seed'] for r in runs}):
        indices=[i for i,r in enumerate(runs) if r['cv_seed']==seed]
        entry={'seed':seed,'folds':len(indices)}
        for name in flattened[0]:
            values=[flattened[i][name] for i in indices]
            entry[name+'_mean']=float(np.mean(values)) if all(v is not None for v in values) else None
            entry[name+'_std']=float(np.std(values,ddof=1)) if all(v is not None for v in values) and len(values)>1 else None
        seed_rows.append(entry)
    write_table(folder/'cv_summary_by_seed.csv',seed_rows)
    if save_plots:
        from matplotlib import pyplot as plt
        epoch=np.array([r['epoch'] for r in curves])
        for metric,title in [('eva_auc','AUC ROC EVA vs resto'),('loss','Loss')]:
            fig,ax=plt.subplots(figsize=(7,4))
            for key,label in [(metric,'Training'),('val_'+metric,'Validation')]:
                avg=np.array([r[key+'_mean'] for r in curves]);spread=np.array([r[key+'_std'] for r in curves])
                line=ax.plot(epoch,avg,label=label+' media')[0]
                lower,upper=avg-spread,avg+spread
                if metric=='eva_auc':lower=np.clip(lower,0,1);upper=np.clip(upper,0,1)
                ax.fill_between(epoch,lower,upper,color=line.get_color(),alpha=.18,label=label+' ±1 SD')
            value=summary['test_eva_auc_roc_mean'] if metric=='eva_auc' else summary['test_loss_mean']
            ax.axhline(value,color='black',linestyle='--',label='Media test dei checkpoint')
            ax.set(xlabel='Epoca (intervallo comune a tutti i run)',ylabel=title,
                   title=f'{title} — media {n_folds} fold, {summary["seeds"]} seed')
            if metric=='eva_auc':ax.set_ylim(0,1.02)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
            finish(fig,folder/f'mean_{metric}_epochs.png',show)
        fig=confusion_figure(mean,labels,'Confusion matrix test — media e SD tra fold',std,normalized=True)
        finish(fig,folder/'mean_confusion_matrix.png',show)
    print('RIEPILOGO CV: EVA F1={:.3f} ± {:.3f}, recall={:.3f} ± {:.3f}'.format(
        summary['test_eva_f1_mean'],summary['test_eva_f1_std'],
        summary['test_eva_recall_mean'],summary['test_eva_recall_std']))
    return summary
