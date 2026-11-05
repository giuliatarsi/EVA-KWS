# %% CONFIGURAZIONE UTENTE (eseguire questa cella per prima, oppure tutto il file con F5)
"""Aprire in Spyder, modificare questa sezione e premere F5.

SINGLE = una configurazione; CUSTOM_GRID = grid selezionata o completa.
Campagna tre classi: grid completa + prove mirate abbinate agli esperimenti binari.
"""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0,str(PROJECT))

from training.config import Settings, MFCCConfig, ModelConfig

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
    learning_rate=0.005,
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
# La verifica mirata mantiene LR 0.005 per isolare il cambio di batch.
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
# Grid completa: 4 MFCC x 24 reti x 1 LR = 96 configurazioni, 480 training/seed.
# Per ridurre le prove, accorciare le liste sopra. Batch e augmentation fissi.
# Queste varianti sono esperimenti, non repliche esatte di architetture pubblicate.

# %% GRID MIRATA: SOLO COPPIE SELEZIONATE, NON UN PRODOTTO CARTESIANO
GRID_SCOPE = 'three_class_campaign'  # 'three_class_campaign', 'selected' oppure 'full'
# ID Gxx = posizione nella grid originale a batch 64, ordinata per F1 EVA.
# Ogni riga associa ESATTAMENTE una configurazione MFCC a una rete.
SELECTED_CONFIGURATIONS = [
    ('G01_best_f1', MFCCConfig(n_mfcc=13, n_mels=32),
     ModelConfig(family='cnn1d', filters=(16,32), kernel=7, dropout=.25)),
    ('G02_compact_k7', MFCCConfig(n_mfcc=13, n_mels=40),
     ModelConfig(family='cnn1d', filters=(8,16), kernel=7, dropout=.25)),
    ('G03_compact_k5', MFCCConfig(n_mfcc=13, n_mels=40),
     ModelConfig(family='cnn1d', filters=(8,16), kernel=5, dropout=.25)),
    ('G04_best_recall', MFCCConfig(n_mfcc=20, n_mels=40),
     ModelConfig(family='cnn1d', filters=(16,32), kernel=5, dropout=.25)),
    ('G05_three_blocks_k5', MFCCConfig(n_mfcc=20, n_mels=40),
     ModelConfig(family='cnn1d', filters=(8,16,32), kernel=5, dropout=.25)),
    ('G07_three_blocks_k3', MFCCConfig(n_mfcc=20, n_mels=32),
     ModelConfig(family='cnn1d', filters=(8,16,32), kernel=3, dropout=.25)),
]
# 6 configurazioni x 1 LR x 5 fold x 1 seed = 30 training.
# Per verificare più seed: SETTINGS.seeds=(42,123,2026) -> 90 training.
# Le CNN2D non sono incluse: nessuna supera questi candidati nella grid a batch 64.

# %% CAMPAGNA A TRE CLASSI
# MODE='custom_grid', GRID_SCOPE='three_class_campaign', SETTINGS.binary=False.
# Il piano replica la ricerca binaria: non si aggiunge una griglia cartesiana di tutto.
# Gli override di batch/LR/dropout/augmentation dei singoli casi sono in campaigns.py.
# 100 epoche; NO early stopping, checkpoint sempre scelto con minima val_loss.
# Le storie numeriche sono conservate per grafici e report; nessuna finestra aperta.
CAMPAIGN_SAVE_PLOTS = False
CAMPAIGN_CONTINUE_ON_ERROR = True  # Registra gli errori e continua; Stop funziona normalmente.

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


def get_selected_configurations():
    """Valida le coppie mirate senza importare TensorFlow né avviare training."""
    if not SELECTED_CONFIGURATIONS:
        raise ValueError('La grid selezionata deve contenere almeno una configurazione')
    names = [name for name, _, _ in SELECTED_CONFIGURATIONS]
    pairs = [(feat, model) for _, feat, model in SELECTED_CONFIGURATIONS]
    if len(set(names)) != len(names) or len(set(pairs)) != len(pairs):
        raise ValueError('Nomi o coppie duplicate nella grid selezionata')
    if any(not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name)
           for name in names):
        raise ValueError('Usare nomi semplici senza spazi o separatori di percorso')
    return SELECTED_CONFIGURATIONS


def main():
    from dataclasses import replace
    import math
    if MODE == 'custom_grid' and GRID_SCOPE == 'three_class_campaign':
        from training.campaigns import run_three_class_campaign
        return run_three_class_campaign(SETTINGS,
                    save_plots=CAMPAIGN_SAVE_PLOTS,
                    continue_on_error=CAMPAIGN_CONTINUE_ON_ERROR)
    if MODE == 'custom_grid' and GRID_SCOPE == 'selected':
        selected = get_selected_configurations()
        stages = [(name, [feat], [model]) for name, feat, model in selected]
        configurations = len(selected)
    else:
        if MODE == 'custom_grid' and GRID_SCOPE != 'full':
            raise ValueError("GRID_SCOPE deve essere 'selected' oppure 'full'")
        features, models = get_configurations()
        stages = [('all', features, models)]
        configurations = len(features)*len(models)
    rates = [SETTINGS.learning_rate] if MODE == 'single' else GRID_LEARNING_RATES
    if not rates or any(not math.isfinite(lr) or lr <= 0 for lr in rates):
        raise ValueError('Learning rate: inserire valori finiti e positivi')
    if len(set(rates)) != len(rates):
        raise ValueError('Learning rate duplicati nella grid')
    folds = SETTINGS.n_folds if SETTINGS.cross_validation else 1
    count = configurations*len(rates)*folds*len(SETTINGS.seeds)
    print(f'Piano: {configurations} configurazioni x {len(rates)} LR '
          f'x {folds} fold x {len(SETTINGS.seeds)} seed = {count} training.')
    print(f'Batch: {SETTINGS.batch_size}; learning rate: {rates}; binary={SETTINGS.binary}.')
    print(f'Epoche: {SETTINGS.epochs}; checkpoint con minima validation loss.')
    from training.experiments import run_search, write_table
    combined = []
    if MODE == 'custom_grid':
        label = 'targeted_grid' if GRID_SCOPE == 'selected' else 'custom_grid'
        grid_root = SETTINGS.output_dir/f'{label}_batch{SETTINGS.batch_size}'
    for rate in rates:
        for name, features, models in stages:
            local = replace(SETTINGS, learning_rate=rate,
                            show_plots=(MODE=='single' and SETTINGS.show_plots),
                            save_plots=(MODE=='single'))
            if MODE == 'custom_grid':
                local.output_dir = grid_root/f'lr_{rate:g}'/name
            if local.cross_validation:
                from training.cross_validation import run_cross_validation
                rows = run_cross_validation(local, features, models,
                                            display_plots=local.show_plots,
                                            save_plots=local.save_plots)
            else:
                rows = run_search(local, features, models)
            if MODE == 'single':
                return rows
            combined.extend(dict(row, grid_learning_rate=rate, candidate=name) for row in rows)
            write_table(grid_root/'grid_summary_all_learning_rates.csv', combined)
    return combined

if __name__=='__main__':
    results=main()  # La variabile results resta consultabile in Spyder.
