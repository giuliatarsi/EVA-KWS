"""Indice WAV e split per gruppi. Il test non partecipa alla scelta della validation."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import re
import wave
import numpy as np
from training.config import digest

LABELS = ('eva', 'noise', 'unknown')
SUBSETS = {'training': 'train', 'train': 'train', 'validation': 'validation',
           'val': 'validation', 'test': 'test', 'test_sviluppo': 'test',
           'test_finale': 'test_finale'}


def index_dataset(root, binary=False):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Cartella dataset assente: {root}. Leggere README.md.')
    records = []
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.suffix.lower() != '.wav' or p.name.startswith('.'):
            continue
        rel = p.relative_to(root)
        if len(rel.parts) < 3 or rel.parts[0] not in SUBSETS or rel.parts[1] not in LABELS:
            raise ValueError(f'Struttura non riconosciuta: {rel}. Attesa training/eva/file.wav ecc.')
        match = re.search(r'(?:^|_)([PA]\d+)(?=_|\.|$)', p.stem, re.I)
        if not match:
            raise ValueError(f'Manca il gruppo Pxx/Axx nel nome: {rel}')
        label = rel.parts[1]
        content = p.read_bytes()
        try:
            with wave.open(str(p), 'rb') as w:
                fmt = (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes())
                audio_hash = hashlib.sha256(w.readframes(w.getnframes())).hexdigest()
        except (wave.Error, EOFError) as exc:
            raise ValueError(f'WAV PCM non leggibile: {rel}') from exc
        records.append(dict(path=str(p), relative=str(rel), original_label=label,
                            label=('non_target' if binary and label != 'eva' else label),
                            group=match.group(1).upper(), subset=SUBSETS[rel.parts[0]],
                            sha256=hashlib.sha256(content).hexdigest(), audio_hash=audio_hash,
                            rate=fmt[0], channels=fmt[1], width=fmt[2], frames=fmt[3]))
    if not records:
        raise ValueError(f'Nessun WAV trovato in {root}')
    _check_overlap(records)
    return records


def _check_overlap(records):
    groups, sounds = {}, {}
    for r in records:
        for key, lookup in [(r['group'], groups), (r['audio_hash'], sounds)]:
            previous = lookup.get(key)
            if previous and previous['subset'] != r['subset']:
                raise ValueError('Data leakage: gruppo o audio identico in insiemi diversi: '
                                 f"{previous['relative']} / {r['relative']}")
            lookup[key] = r


def split_validation(records, fraction=.2, seed=42):
    """Usa validation esplicita; altrimenti ricerca riproducibile su soli gruppi train.

    Approssima quota e distribuzione classi. Non ottimizza usando risultati ML.
    """
    records = [dict(r) for r in records]
    if any(r['subset'] == 'validation' for r in records):
        _validate_classes(records)
        return records
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction deve essere tra 0 e 1')
    train = [r for r in records if r['subset'] == 'train']
    groups = sorted({r['group'] for r in train})
    labels = sorted({r['original_label'] for r in train})
    if len(groups) < 2:
        raise ValueError('Servono più gruppi per creare validation senza leakage')
    counts = np.array([[sum(r['group'] == g and r['original_label'] == c for r in train)
                        for c in labels] for g in groups], dtype=float)
    totals = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    best = None
    # Prova anche cardinalità vicine: quote esatte non sempre possibili per gruppi.
    for _ in range(4000):
        k = max(1, min(len(groups)-1, round(len(groups)*fraction) + int(rng.integers(-1, 2))))
        chosen = rng.choice(len(groups), k, replace=False)
        val = counts[chosen].sum(axis=0)
        if np.any(val == 0) or np.any(val == totals):
            continue
        score = np.mean((val / totals - fraction)**2)
        score += (val.sum()/totals.sum() - fraction)**2
        if best is None or score < best[0]:
            best = (score, {groups[i] for i in chosen})
    if best is None:
        raise ValueError('Impossibile avere tutte le classi in train/validation separando i gruppi. '
                         'Servono altri gruppi indipendenti, oppure cartella validation esplicita.')
    for r in records:
        if r['subset'] == 'train' and r['group'] in best[1]:
            r['subset'] = 'validation'
    _check_overlap(records)
    _validate_classes(records)
    return records


def _validate_classes(records):
    labels = {r['label'] for r in records if r['subset'] == 'train'}
    for subset in ['train', 'validation']:
        if {r['label'] for r in records if r['subset'] == subset} != labels or len(labels) < 2:
            raise ValueError(f'Classi mancanti in {subset}')


def fingerprint(records):
    return digest([{k:r[k] for k in ['relative','sha256','label','group','subset']} for r in records])


def save_index(records, path):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader(); writer.writerows(records)


def describe(records):
    for subset in ['train','validation','test','test_finale']:
        selected = [r for r in records if r['subset'] == subset]
        print(subset, dict(Counter(r['label'] for r in selected)),
              'gruppi:', ', '.join(sorted({r['group'] for r in selected})))


def training_class_weights(y,n_classes,mode='none'):
    """Pesi N/(K*n_c), senza usare validation/test; media pesata sui campioni = 1."""
    y=np.asarray(y,dtype=int)
    if mode not in ('none','balanced'):raise ValueError('class_weight_mode: usare none o balanced')
    counts=np.bincount(y,minlength=n_classes)
    if len(counts)!=n_classes or np.any(counts==0):raise ValueError('Classi mancanti o indici errati nel training')
    weights=None if mode=='none' else {i:float(len(y)/(n_classes*counts[i])) for i in range(n_classes)}
    return weights,counts.tolist()
