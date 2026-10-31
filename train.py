# %% CONFIGURAZIONE UTENTE (eseguire questa cella per prima, oppure tutto il file con F5)
"""Aprire in Spyder, modificare questa sezione e premere F5.

SINGLE = una configurazione; CUSTOM_GRID = FEATURES x MODELS x GRID_LEARNING_RATES.
Grid binaria: 192 configurazioni, 960 training con 5 fold/un seed.
"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))

from config import Settings, MFCCConfig, ModelConfig

MODE = 'single'  # Scegliere SOLO 'single' oppure 'custom_grid'
SETTINGS = Settings(
    data_dir=PROJECT/'data_v2',       # Sostituire con la cartella dei WAV da 1100 ms
    output_dir=PROJECT/'results',
    cache_dir=PROJECT/'cache',
    cross_validation=True,        # Riunisce train/test e fa CV per gruppi
    n_folds=5,                   # Ogni gruppo è test una volta
    show_plots=True,             # In single; nelle grid nessuna finestra grafica
    class_weight_mode='none',    # 'balanced': pesi di classe dal training di ogni fold
    binary=True,                 # True: unknown + noise -> non_target
    seeds=(42,),                  # (42, 123, 2026) per tre ripetizioni
    epochs=100,
    batch_size=128,
    learning_rate=0.004,
    early_stopping=False,        # 100 epoche per tutti; resta il checkpoint migliore
    patience=15,
    gaussian_std=0.45,
    time_mask_max_frames=1,
)
SINGLE_FEATURE = MFCCConfig(n_mfcc=13,n_mels=40)
SINGLE_MODEL = ModelConfig(family='cnn1d',filters=(8,16),kernel=5,dropout=.25)

# %% GRID PERSONALIZZABILE
# Per avviare: impostare MODE = 'custom_grid' in alto, poi F5.
# SINGLE usa SETTINGS.learning_rate; la grid usa SOLO GRID_LEARNING_RATES.
# Valori richiesti 0.01 e 0.05: NON equivalgono a 0.001 e 0.005.
# Sono superiori ai valori 0.004/0.005 provati finora.
GRID_LEARNING_RATES = [0.005]
# Per confrontare invece i valori precedenti: [0.004, 0.005]

FEATURES = [
    MFCCConfig(n_mfcc=13, n_mels=32),  # Riferimento
    MFCCConfig(n_mfcc=20, n_mels=32),  # Più coefficienti
    MFCCConfig(n_mfcc=13, n_mels=40),  # Più filtri Mel
    MFCCConfig(n_mfcc=20, n_mels=40),  # Più coefficienti e filtri Mel
]
# Tutte: audio 16 kHz, clip 1100 ms, frame/hop 20 ms, FFT256, CMVNW101.
# DSP EI versione 4 assunta; da verificare con l'export del proprio progetto.

GRID_FAMILIES = ['cnn1d', 'cnn2d']
GRID_KERNELS = [3, 5, 7]  # Per CNN2D: 3x3, 5x5, 7x7
GRID_FILTERS = [
    (4, 8),       # Due blocchi, rete piccola
    (8, 16),      # Due blocchi, riferimento
    (16, 32),     # Due blocchi, rete più larga
    (8, 16, 32),  # Tre blocchi, rete più profonda
]
# Prodotto completo: 2 famiglie x 3 kernel x 4 tuple = 24 reti.
# Ogni blocco: convoluzione ReLU, max pooling, dropout 0.25; testa Flatten.
MODELS = [
    ModelConfig(family=family, filters=filters, kernel=kernel, dropout=.25)
    for family in GRID_FAMILIES
    for filters in GRID_FILTERS
    for kernel in GRID_KERNELS
]
# 4 MFCC x 24 reti x 2 LR = 192 configurazioni; x 5 fold = 960 training/seed.
# Per ridurre le prove, accorciare le liste sopra. Batch e augmentation fissi.
# Queste varianti sono esperimenti, non repliche esatte di architetture pubblicate.

# %% ESECUZIONE

def get_configurations(mode=None):
    """Restituisce il piano senza importare TensorFlow o avviare training."""
    mode = MODE if mode is None else mode
    if mode == 'single':
        return [SINGLE_FEATURE], [SINGLE_MODEL]
    if mode == 'custom_grid':
        if not FEATURES or not MODELS:
            raise ValueError('FEATURES e MODELS devono contenere almeno un elemento')
        if len(set(FEATURES)) != len(FEATURES) or len(set(MODELS)) != len(MODELS):
            raise ValueError('Configurazioni duplicate nelle liste: rimuoverle prima di lanciare la grid')
        return FEATURES, MODELS
    raise ValueError("MODE deve essere 'single' oppure 'custom_grid'")


def main():
    from dataclasses import replace
    import math
    features, models = get_configurations()
    rates = [SETTINGS.learning_rate] if MODE == 'single' else GRID_LEARNING_RATES
    if not rates or any(not math.isfinite(lr) or lr <= 0 for lr in rates):
        raise ValueError('Learning rate: inserire valori finiti e positivi')
    if len(set(rates)) != len(rates):
        raise ValueError('Learning rate duplicati nella grid')
    folds = SETTINGS.n_folds if SETTINGS.cross_validation else 1
    count = len(features)*len(models)*len(rates)*folds*len(SETTINGS.seeds)
    print(f'Piano: {len(features)} MFCC x {len(models)} reti x {len(rates)} LR '
          f'x {folds} fold x {len(SETTINGS.seeds)} seed = {count} training.')
    print(f'Learning rate: {rates}; binary={SETTINGS.binary}.')
    print(f'Epoche: {SETTINGS.epochs}; checkpoint con minima validation loss.')
    from experiments import run_search, write_table
    combined = []
    for rate in rates:
        local = replace(SETTINGS, learning_rate=rate,
                        show_plots=(MODE=='single' and SETTINGS.show_plots),
                        save_plots=(MODE=='single'))
        if MODE == 'custom_grid':
            # Cartelle separate: un LR non sovrascrive i riepiloghi dell'altro.
            local.output_dir = SETTINGS.output_dir/'custom_grid'/f'lr_{rate:g}'
        if local.cross_validation:
            from cross_validation import run_cross_validation
            rows = run_cross_validation(local, features, models,
                                        display_plots=local.show_plots,
                                        save_plots=local.save_plots)
        else:
            rows = run_search(local, features, models)
        if MODE == 'single':
            return rows
        combined.extend(dict(row, grid_learning_rate=rate) for row in rows)
        write_table(SETTINGS.output_dir/'custom_grid'/'grid_summary_all_learning_rates.csv', combined)
    return combined

if __name__=='__main__':
    results=main()  # La variabile results resta consultabile in Spyder.
