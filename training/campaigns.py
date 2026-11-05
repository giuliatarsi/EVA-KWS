"""Campagna a tre classi abbinata agli esperimenti binari: piano esplicito e deduplicato."""
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
import csv
import json
import math
from training.config import MFCCConfig, ModelConfig, plain, digest


@dataclass(frozen=True)
class CampaignCase:
    name: str
    phases: tuple
    feature: MFCCConfig
    model: ModelConfig
    batch_size: int = 64
    learning_rate: float = .005
    gaussian_std: float = .45
    time_mask_max_frames: int = 1
    class_weight_mode: str = 'none'
    early_stopping: bool = False

    def training_overrides(self):
        return {key: getattr(self, key) for key in
                ('batch_size', 'learning_rate', 'gaussian_std',
                 'time_mask_max_frames', 'class_weight_mode', 'early_stopping')}

    def signature(self):
        return {'mfcc': plain(self.feature), 'model': plain(self.model),
                'training': self.training_overrides()}

    @property
    def key(self):
        return digest(self.signature())[:16]


def build_three_class_campaign():
    """Replica le condizioni binarie; i casi coincidenti conservano tutti i tag di fase."""
    cases = {}

    def add(name, phase, c=13, m=40, family='cnn1d', filters=(8, 16),
            kernel=5, dropout=.25, **training):
        case = CampaignCase(name, (phase,), MFCCConfig(n_mfcc=c, n_mels=m),
                            ModelConfig(family=family, filters=filters,
                                        kernel=kernel, dropout=dropout), **training)
        if case.key in cases:
            old = cases[case.key]
            cases[case.key] = replace(old, phases=tuple(dict.fromkeys(old.phases + case.phases)))
        else:
            cases[case.key] = case

    # Identico disegno della grid binaria a batch 64: 4 x 2 x 4 x 3 = 96.
    for (c, m), family, filters, kernel in product(
            [(13, 32), (20, 32), (13, 40), (20, 40)],
            ['cnn1d', 'cnn2d'], [(4, 8), (8, 16), (16, 32), (8, 16, 32)], [3, 5, 7]):
        add(f'grid_{family}_f{"-".join(map(str, filters))}_k{kernel}_mfcc{c}_mel{m}',
            'architecture_grid_batch64', c, m, family, filters, kernel)

    # Tutte le condizioni della ablation binaria, stesso front-end/candidato/batch.
    for name, dropout, std, mask in [
            ('none', 0., 0., 0), ('augmentation_only', 0., .45, 1),
            ('dropout_only', .25, 0., 0), ('both_d025', .25, .45, 1),
            ('both_d050', .50, .45, 1)]:
        add('ablation_' + name, 'dropout_augmentation', dropout=dropout,
            gaussian_std=std, time_mask_max_frames=mask)

    for c, m in [(8, 32), (10, 32), (13, 32), (13, 40), (13, 48)]:
        add(f'feature_mfcc{c}_mel{m}', 'mfcc_followup', c=c, m=m)

    # Kernel a front-end costante; già presenti nella grid e non ripetuti.
    for kernel in [3, 5, 7]:
        add(f'kernel_{kernel}', 'kernel_followup', kernel=kernel)

    for lr in [.0025, .004, .005]:
        add(f'lr_mfcc13_mel32_{lr:g}', 'learning_rate_mfcc13_mel32', m=32,
            learning_rate=lr)

    for batch in [32, 64, 128, 256]:
        add(f'batch_{batch}', 'batch_followup', batch_size=batch)
    for lr in [.004, .005]:
        add(f'lr_batch128_{lr:g}', 'learning_rate_batch128', batch_size=128,
            learning_rate=lr)

    # Pesi di classe disattivati: nessuna prova balanced nella campagna.

    # Stesse sei coppie della grid mirata binaria, senza altri incroci.
    selected = [
        ('G01', 13, 32, (16, 32), 7), ('G02', 13, 40, (8, 16), 7),
        ('G03', 13, 40, (8, 16), 5), ('G04', 20, 40, (16, 32), 5),
        ('G05', 20, 40, (8, 16, 32), 5), ('G07', 20, 32, (8, 16, 32), 3),
    ]
    for name, c, m, filters, kernel in selected:
        add('targeted_' + name, 'targeted_candidates_batch128', c=c, m=m,
            filters=filters, kernel=kernel, batch_size=128)
    return list(cases.values())


def _write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Campi uniti: permette metriche di confronto aggiunte in un secondo momento.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')


def plan_rows(cases):
    return [dict(case_id=c.key, name=c.name, phases=';'.join(c.phases),
                 family=c.model.family, filters='-'.join(map(str, c.model.filters)),
                 kernel=c.model.kernel, dropout=c.model.dropout,
                 n_mfcc=c.feature.n_mfcc, n_mels=c.feature.n_mels,
                 **c.training_overrides()) for c in cases]


def _preflight(settings, destination):
    """Verifica le assegnazioni su entrambi i task senza importare TensorFlow."""
    from training.data import index_dataset
    from training.cross_validation import make_folds, save_fold_assignment
    three = index_dataset(settings.data_dir, binary=False)
    binary = index_dataset(settings.data_dir, binary=True)
    kwargs = dict(n_splits=settings.n_folds, seed=settings.split_seed,
                  validation_fraction=settings.validation_fraction)
    three_folds, assignment = make_folds(three, **kwargs)
    binary_folds, _ = make_folds(binary, **kwargs)
    identity = lambda fold: sorted((r['relative'], r['subset'], r['audio_hash']) for r in fold)
    if any(identity(a) != identity(b) for a, b in zip(three_folds, binary_folds)):
        raise ValueError('I fold binari e a tre classi differiscono: confronto sospeso')
    # Verifica tutte le classi in ogni sottoinsieme, non solo sul pool.
    for fold in three_folds:
        for subset in ['train', 'validation', 'test']:
            if {r['label'] for r in fold if r['subset'] == subset} != {'eva', 'noise', 'unknown'}:
                raise ValueError(f'Mancano classi nel sottoinsieme {subset}')
    save_fold_assignment(three, three_folds, assignment, destination / 'splits')
    certificate = dict(same_audio_and_assignments_across_tasks=True,
                       samples=len(three), n_folds=len(three_folds),
                       split_seed=settings.split_seed,
                       validation_fraction=settings.validation_fraction,
                       original_class_counts={label: sum(r['original_label'] == label for r in three)
                                              for label in ['eva', 'noise', 'unknown']})
    _save_json(destination / 'split_comparison.json', certificate)
    return certificate


def _binary_references(results_dir, settings):
    """Indicizza soltanto esperimenti binari già conclusi, mai usati per il training."""
    roots = [Path(results_dir) / name for name in
             ['custom_grid', 'cross_validation', 'targeted_grid_batch128']]
    references = {}
    for root in roots:
        for source in root.rglob('cv_summary.json'):
            configs = sorted(source.parent.glob('seed_*/fold_01/*/config.json'))
            if not configs:
                continue
            cfg = json.loads(configs[-1].read_text())
            st = cfg['settings']
            if not st.get('binary'):
                continue
            common = ['epochs', 'split_seed', 'validation_fraction', 'n_folds', 'seeds', 'patience']
            if any(plain(st.get(k)) != plain(getattr(settings, k)) for k in common):
                continue
            overrides = {k: st[k] for k in CampaignCase('x', (), MFCCConfig(), ModelConfig()).training_overrides()}
            key = digest(dict(mfcc=cfg['mfcc'], model=cfg['model'], training=overrides))[:16]
            ref = dict(source_summary=str(source), source_folder=str(source.parent),
                       metrics=json.loads(source.read_text()))
            if key not in references or source.stat().st_mtime > Path(references[key]['source_summary']).stat().st_mtime:
                references[key] = ref
    return references


def _paired_comparison(three_row, reference, settings):
    """Confronto EVA-vs-rest; esclude il confronto diretto fra accuracy di task diversi."""
    folder = Path(three_row['summary_folder'])
    other = Path(reference['source_folder'])
    for seed in settings.seeds:
        for fold in range(1, settings.n_folds + 1):
            def mapping(root):
                files = list(root.glob(f'seed_{seed}/fold_{fold:02d}/*/manifest.csv'))
                if not files:
                    raise ValueError(f'Manifest mancante in {root}')
                maps = []
                for file in files:
                    with file.open() as f:
                        maps.append(sorted((x['relative'], x['subset'], x['audio_hash'])
                                           for x in csv.DictReader(f)))
                if any(m != maps[0] for m in maps):
                    raise ValueError('Manifest multipli non equivalenti')
                return maps[0]
            if mapping(folder) != mapping(other):
                raise ValueError('Dataset/split del riferimento binario differente')
    result = dict(case_id=three_row['case_id'], phases=three_row['phases'],
                  same_audio_and_splits=True, binary_summary=reference['source_summary'])
    binary = reference['metrics']
    for metric in ['precision', 'recall', 'f1', 'auc_roc', 'false_positive_rate', 'fp', 'fn']:
        key = 'test_eva_' + metric + '_mean'
        result['binary_' + metric] = binary.get(key)
        result['three_class_' + metric] = three_row.get(key)
        if binary.get(key) is not None and three_row.get(key) is not None:
            result['delta_three_minus_binary_' + metric] = three_row[key] - binary[key]
    # Differenze delle medie: le somme OOF (seeds multipli) si consultano nei report per seed.
    # Accuracy comparable: collapse noise/unknown into non_target for this metric only.
    # A confusion between the two negative classes does not count as an EVA detection error.
    for task, root in [('binary', other), ('three_class', folder)]:
        accuracies = []
        for seed in settings.seeds:
            for fold in range(1, settings.n_folds + 1):
                reports = list(root.glob(f'seed_{seed}/fold_{fold:02d}/*/test_metrics.json'))
                report = json.loads(reports[0].read_text())
                eva = report['per_class']['eva']
                total = sum(eva[k] for k in ['tp', 'tn', 'fp', 'fn'])
                accuracies.append((eva['tp'] + eva['tn']) / total)
        result[task + '_eva_vs_rest_accuracy_mean'] = sum(accuracies) / len(accuracies)
    result['binary_parameters'] = binary.get('parameters')
    result['three_class_parameters'] = three_row.get('parameters')
    return result


def run_three_class_campaign(settings, cases=None, save_plots=False, continue_on_error=True):
    """Esegue solo il task a tre classi e conserva storie/metriche per il report successivo."""
    if settings.binary or not settings.cross_validation:
        raise ValueError('La campagna richiede binary=False e cross_validation=True')
    cases = build_three_class_campaign() if cases is None else list(cases)
    if not cases or len({c.key for c in cases}) != len(cases):
        raise ValueError('Il piano è vuoto o contiene esperimenti duplicati')
    if any(not math.isfinite(c.learning_rate) or c.learning_rate <= 0 for c in cases):
        raise ValueError('Learning rate non valido')
    base = Path(settings.output_dir) / 'three_class_campaign'
    base.mkdir(parents=True, exist_ok=True)
    _write_csv(base / 'experiment_plan.csv', plan_rows(cases))
    _save_json(base / 'experiment_plan.json', dict(settings=plain(settings),
               experiments=[dict(id=c.key, name=c.name, phases=c.phases, **c.signature()) for c in cases]))
    _preflight(settings, base)
    references = _binary_references(settings.output_dir, settings)
    _save_json(base / 'binary_reference_index.json', references)
    print(f'Campagna: {len(cases)} configurazioni uniche, '
          f'{len(cases)*settings.n_folds*len(settings.seeds)} training. Output: {base}', flush=True)
    print('Classi EVA/noise/unknown; confronto binario disponibile per',
          sum(c.key in references for c in cases), 'configurazioni.', flush=True)
    from training.cross_validation import run_cross_validation
    summary = []; comparisons = []; failures = []
    for i, case in enumerate(cases, 1):
        local = replace(settings, **case.training_overrides(), binary=False,
                        output_dir=base / f'{i:03d}_{case.name}_{case.key}',
                        show_plots=False, save_plots=save_plots)
        print(f'\nCAMPAGNA {i}/{len(cases)}: {case.name}; fasi={case.phases}', flush=True)
        try:
            rows = run_cross_validation(local, [case.feature], [case.model],
                                        display_plots=False, save_plots=save_plots)
            row = dict(rows[0], case_id=case.key, campaign_name=case.name,
                       phases=';'.join(case.phases), binary=False,
                       gaussian_std=case.gaussian_std,
                       time_mask_max_frames=case.time_mask_max_frames,
                       early_stopping=case.early_stopping)
            summary.append(row)
            _write_csv(base / 'campaign_summary.csv', summary)
        except Exception as exc:
            # KeyboardInterrupt/SystemExit non sono catturati: Stop interrompe la campagna.
            failures.append(dict(case_id=case.key, name=case.name, error=repr(exc)))
            _save_json(base / 'failed_experiments.json', failures)
            if not continue_on_error:
                raise
            print('Esperimento fallito, passo al successivo:', repr(exc), flush=True)
            continue
        if case.key in references:
            try:
                comparisons.append(_paired_comparison(row, references[case.key], settings))
                _write_csv(base / 'binary_vs_three_class.csv', comparisons)
            except Exception as exc:
                failures.append(dict(case_id=case.key, name=case.name,
                                     comparison_error=repr(exc)))
                _save_json(base / 'failed_experiments.json', failures)
                print('Confronto binario non valido:', repr(exc), flush=True)
    _save_json(base / 'campaign_status.json', dict(planned=len(cases), completed=len(summary),
                failures=failures, binary_comparisons=len(comparisons)))
    return summary
