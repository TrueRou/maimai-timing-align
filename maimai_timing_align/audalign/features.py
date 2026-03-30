from pathlib import Path

import librosa
import numpy as np

from .models import AudioAlignParams


def load_audio_mono(path: Path, params: AudioAlignParams) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(
        str(path),
        sr=int(params.sample_rate),
        mono=True,
        duration=float(params.max_duration_sec) if params.max_duration_sec is not None else None,
    )
    if y.size < 128:
        raise RuntimeError(f"音频过短或无法读取: {path}")
    return y.astype(np.float32), int(sr)


def extract_onset_envelope(y: np.ndarray, sr: int, params: AudioAlignParams) -> np.ndarray:
    env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=int(params.hop_length),
        n_fft=int(params.n_fft),
        aggregate=np.median,
    )
    return np.asarray(env, dtype=np.float64)
