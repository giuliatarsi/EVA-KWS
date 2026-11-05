"""MFCC dal sorgente ufficiale EI; parità con il progetto da verificare su WAV/reference."""
from pathlib import Path
import wave
import numpy as np
from training.config import plain, digest


def read_clip(record, cfg, tolerance_ms=1.0):
    with wave.open(record['path'], 'rb') as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getcomptype()) != (cfg.sample_rate, 1, 2, 'NONE'):
            raise ValueError(f"Richiesto WAV PCM16 mono {cfg.sample_rate} Hz: {record['relative']}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(np.float32)
    expected = round(cfg.sample_rate * cfg.clip_ms / 1000)
    if abs(len(x)-expected) > cfg.sample_rate*tolerance_ms/1000:
        raise ValueError(f"{record['relative']}: {len(x)/cfg.sample_rate*1000:.2f} ms; attesi {cfg.clip_ms}. "
                         'Ritagliare prima gli audio: nessun taglio automatico della frase.')
    return np.pad(x[:expected], (0, max(0, expected-len(x))))


def mfcc(signal, cfg):
    """Replica generate_features del blocco MFCC ufficiale (senza grafici).

    Input: valori PCM16 in float, senza divisione per 32768.
    Non aggiunge normalizzazione globale di training.
    """
    from training.third_party.ei_speechpy import feature, processing
    if cfg.implementation != 'edgeimpulse_9b0ceb2b5d22_speechpy_v1':
        raise ValueError('Cache/configurazione DSP precedente: usare MFCCConfig aggiornato')
    if not 1 <= cfg.n_mfcc <= cfg.n_mels or cfg.n_mels < 2:
        raise ValueError('Richiesto 1 <= n_mfcc <= n_mels e almeno 2 filtri')
    if cfg.dsp_version not in (1,2,3,4):
        raise ValueError('dsp_version deve essere 1, 2, 3 o 4')
    if cfg.n_fft < 2 or cfg.n_fft & (cfg.n_fft-1):
        raise ValueError('FFT deve essere una potenza di due')
    if cfg.normalization_window < 0 or (cfg.normalization_window and cfg.normalization_window % 2 == 0):
        raise ValueError('normalization_window deve essere zero oppure dispari')
    high = cfg.high_hz or cfg.sample_rate/2
    if not 0 <= cfg.low_hz < high <= cfg.sample_rate/2:
        raise ValueError('Limiti frequenza non validi')
    x = np.asarray(signal, dtype=float)
    count,_,_ = processing.calculate_number_of_frames(
        x,cfg.sample_rate,cfg.dsp_version,cfg.frame_ms/1000,cfg.hop_ms/1000,False)
    if not 1 <= count <= 500:
        raise ValueError('Numero di frame fuori dai limiti del blocco EI (1..500)')
    pre = processing.preemphasis(x,cof=cfg.preemphasis,shift=cfg.pre_shift)
    result = feature.mfcc(pre,sampling_frequency=cfg.sample_rate,
        implementation_version=cfg.dsp_version,frame_length=cfg.frame_ms/1000,
        frame_stride=cfg.hop_ms/1000,num_cepstral=cfg.n_mfcc,num_filters=cfg.n_mels,
        fft_length=cfg.n_fft,low_frequency=cfg.low_hz,
        high_frequency=None if cfg.high_hz==0 else cfg.high_hz,
        use_old_mels=cfg.dsp_version<=3)
    if cfg.normalization_window:
        result = processing.cmvnw(result,win_size=cfg.normalization_window,variance_normalization=True)
    return result.astype(np.float32)


def feature_matrix(records, cfg, cache_dir, tolerance_ms=1.0):
    cache = Path(cache_dir); cache.mkdir(parents=True, exist_ok=True)
    result = []
    for r in records:
        # Valida anche quando esiste la cache.
        clip = read_clip(r, cfg, tolerance_ms)
        key = digest({'sha256':r['sha256'], 'mfcc':plain(cfg)})
        p = cache/f'{key}.npy'
        if p.exists():
            feat = np.load(p, allow_pickle=False)
        else:
            feat = mfcc(clip, cfg)
            np.save(p, feat, allow_pickle=False)
        if not np.isfinite(feat).all():
            raise ValueError(f"Feature non finite: {r['relative']}")
        result.append(feat)
    if not result:
        raise ValueError('Nessun campione nel sottoinsieme selezionato')
    return np.stack(result)
