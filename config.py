"""Configurazione condivisa. Modificare soprattutto train.py per l'uso in Spyder."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import hashlib
import json

PROJECT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class MFCCConfig:
    sample_rate: int = 16000
    clip_ms: int = 1100
    frame_ms: int = 20
    hop_ms: int = 20
    n_mfcc: int = 13
    n_mels: int = 32
    n_fft: int = 256
    low_hz: float = 0.0
    high_hz: float = 0.0  # 0 = Nyquist, come Edge Impulse
    preemphasis: float = 0.98
    pre_shift: int = 1
    normalization_window: int = 101
    dsp_version: int = 4  # Verificare sul blocco MFCC del progetto esportato
    implementation: str = 'edgeimpulse_9b0ceb2b5d22_speechpy_v1'

@dataclass(frozen=True)
class ModelConfig:
    family: str = 'cnn1d'  # cnn1d oppure cnn2d
    filters: tuple = (8, 16)
    kernel: int = 3
    dropout: float = 0.25
    head: str = 'flatten'  # oppure gap: global average pooling

@dataclass
class Settings:
    data_dir: Path = PROJECT / 'data_v2'
    output_dir: Path = PROJECT / 'results'
    cache_dir: Path = PROJECT / 'cache'
    cross_validation: bool = False
    n_folds: int = 5
    save_plots: bool = True
    show_plots: bool = True
    class_weight_mode: str = 'none'  # none oppure balanced, stimato SOLO sul training
    binary: bool = False
    validation_fraction: float = 0.20
    split_seed: int = 42
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.005
    early_stopping: bool = False  # Tutte le epoche; test sul checkpoint migliore
    patience: int = 15
    gaussian_std: float = 0.45
    time_mask_max_frames: int = 1  # zero per disattivare
    seeds: tuple = (42,)
    deterministic: bool = True
    # False: rifiuta audio diversi da mono 16 kHz PCM16 di 1100 ms.
    # Nessun ricampionamento/ritaglio automatico dei WAV.
    length_tolerance_ms: float = 1.0


def plain(value):
    if hasattr(value, '__dataclass_fields__'):
        return plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), indent=2, allow_nan=False), encoding='utf-8')
