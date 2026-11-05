"""Metriche su clip: FPR = FP / negativi, NON falsi allarmi/ora."""
from pathlib import Path
import csv
import numpy as np
from training.config import write_json
from sklearn.metrics import roc_auc_score, roc_curve


def classification_metrics(y, scores, labels):
    y = np.asarray(y, dtype=int)
    prediction = np.asarray(scores).argmax(axis=1)
    cm = np.zeros((len(labels),len(labels)), dtype=int)
    np.add.at(cm, (y,prediction), 1)
    per_class = {}
    for i, label in enumerate(labels):
        tp = int(cm[i,i]); fn = int(cm[i].sum()-tp); fp = int(cm[:,i].sum()-tp)
        tn = int(cm.sum()-tp-fn-fp)
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        per_class[label] = dict(precision=precision, recall=recall,
                               f1=2*precision*recall/(precision+recall) if precision+recall else 0.0,
                               support=int(cm[i].sum()), tp=tp, fn=fn, fp=fp, tn=tn,
                               false_positive_rate=fp/(fp+tn) if fp+tn else 0.0)
    for i, label in enumerate(labels):
        binary=(y==i).astype(int)
        if len(np.unique(binary))!=2:
            per_class[label]['auc_roc']=None
        else:
            per_class[label]['auc_roc']=float(roc_auc_score(binary,scores[:,i]))
    loss = float(-np.log(np.clip(scores[np.arange(len(y)),y], 1e-7,1)).mean())
    averages = {}
    for metric in ['precision','recall','f1']:
        values=[per_class[l][metric] for l in labels]
        averages['macro_'+metric]=float(np.mean(values))
        averages['weighted_'+metric]=float(np.average(values,weights=cm.sum(axis=1)))
    return dict(accuracy=float(np.mean(y == prediction)), loss=loss, averages=averages, labels=list(labels),
                confusion_matrix=cm.tolist(), per_class=per_class)


def save_evaluation(folder, prefix, records, y, scores, labels):
    folder = Path(folder)
    report = classification_metrics(y, scores, labels)
    write_json(folder/f'{prefix}_metrics.json', report)
    with (folder/f'{prefix}_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        fields = ['file','group','true_label','prediction'] + [f'score_{l}' for l in labels]
        writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r, yi, score in zip(records,y,scores):
            row = dict(file=r['relative'], group=r['group'], true_label=labels[yi],
                       prediction=labels[int(np.argmax(score))])
            row.update({f'score_{l}':float(score[i]) for i,l in enumerate(labels)})
            writer.writerow(row)
    return report


def plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1,2, figsize=(10,4))
    for ax, metric in zip(axes,['loss','accuracy']):
        for key in [metric, 'val_'+metric]:
            ax.plot(range(1,len(history[key])+1), history[key],label=key)
        ax.set(xlabel='Epoch',ylabel=metric);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)
