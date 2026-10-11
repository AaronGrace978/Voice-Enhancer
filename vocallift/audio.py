"""Audio loading, saving, and frame loudness."""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
HOP_LENGTH = 512

_MAGIC = (
    (b"RIFF", ".wav"),
    (b"fLaC", ".flac"),
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
    (b"\x1aE\xdf\xa3", ".webm"),
)


def _suffix_for(data: bytes, filename: str | None = None) -> str:
    if filename:
        ext = Path(filename).suffix.lower()
        if ext:
            return ".wav" if ext == ".wave" else ext
    for magic, ext in _MAGIC:
        if data.startswith(magic):
            return ext
    if len(data) > 1 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return ".mp3"
    return ".wav"


def _channels_last(y: np.ndarray) -> np.ndarray:
    """Return (samples, channels). Accept mono, soundfile, or librosa layouts."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        return y[:, None]
    if y.shape[0] <= 8 and y.shape[1] > y.shape[0]:
        y = y.T
    return y


def _read_array(y: np.ndarray, file_sr: int, sr: int) -> np.ndarray:
    y = _channels_last(y).mean(axis=1)
    if file_sr != sr:
        import librosa

        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return np.ascontiguousarray(y, dtype=np.float32)


def _read_with_librosa(path: str, sr: int) -> np.ndarray:
    import librosa

    y, file_sr = librosa.load(path, sr=None, mono=False)
    return _read_array(y, file_sr, sr)


def load_audio(source: str | Path | bytes, sr: int = SAMPLE_RATE, filename: str | None = None) -> np.ndarray:
    """Load a file or a byte string as mono float32 at ``sr``."""
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        try:
            y, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            return _read_array(y, file_sr, sr)
        except Exception:
            suffix = _suffix_for(data, filename)
            with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                tmp.write(data)
                tmp.flush()
                return _read_with_librosa(tmp.name, sr)
    path = str(source)
    try:
        y, file_sr = sf.read(path, dtype="float32", always_2d=True)
        return _read_array(y, file_sr, sr)
    except Exception:
        return _read_with_librosa(path, sr)


def save_audio(path: str | Path, y: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    sf.write(str(path), np.clip(y, -1.0, 1.0), sr, subtype="PCM_16")


def to_wav_bytes(y: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def frame_loudness_db(y: np.ndarray, hop: int = HOP_LENGTH, n_frames: int | None = None) -> np.ndarray:
    """RMS loudness in dBFS for each hop-sized frame."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if n_frames is None:
        n_frames = max(1, int(np.ceil(len(y) / max(hop, 1))))
    if n_frames <= 0 or len(y) == 0:
        return np.zeros(max(n_frames, 0), dtype=np.float32)
    energy = np.empty(len(y) + 1, dtype=np.float64)
    energy[0] = 0.0
    np.cumsum(y * y, out=energy[1:])
    centers = np.arange(n_frames) * hop
    starts = np.clip(centers - hop, 0, len(y))
    ends = np.clip(centers + hop, 0, len(y))
    counts = np.maximum(ends - starts, 1)
    rms = np.sqrt((energy[ends] - energy[starts]) / counts + 1e-12)
    return (20.0 * np.log10(rms)).astype(np.float32)


def peak_normalize(y: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y))) + 1e-9
    return y * (10 ** (peak_db / 20.0) / peak)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y[:n]
