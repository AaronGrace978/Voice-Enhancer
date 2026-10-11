"""Audio loading, saving, and frame loudness."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
HOP_LENGTH = 512


def resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio. soxr when it is installed, librosa otherwise."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if orig_sr == target_sr or y.size == 0:
        return np.ascontiguousarray(y, dtype=np.float32)
    try:
        import soxr

        out = soxr.resample(y, orig_sr, target_sr, quality="HQ")
    except ImportError:
        import librosa

        out = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    return np.ascontiguousarray(out, dtype=np.float32)


def _decode_av(source, sr: int) -> np.ndarray:
    """Decode mp3, m4a, webm, and anything else soundfile declines."""
    import av

    container = av.open(source)
    try:
        stream = next((item for item in container.streams if item.type == "audio"), None)
        if stream is None:
            raise ValueError("That file has no audio stream.")
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks: list[np.ndarray] = []

        def take(frames) -> None:
            if frames is None:
                return
            if not isinstance(frames, list):
                frames = [frames]
            for frame in frames:
                chunks.append(np.asarray(frame.to_ndarray(), dtype=np.float32).reshape(-1))

        for frame in container.decode(stream):
            take(resampler.resample(frame))
        take(resampler.resample(None))
    finally:
        container.close()
    if not chunks:
        raise ValueError("That recording is empty.")
    return np.concatenate(chunks)


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
    return resample(y, file_sr, sr)


def _load_decoded(source, sr: int) -> np.ndarray:
    try:
        if isinstance(source, (bytes, bytearray)):
            y, file_sr = sf.read(io.BytesIO(bytes(source)), dtype="float32", always_2d=True)
        else:
            y, file_sr = sf.read(str(source), dtype="float32", always_2d=True)
        return _read_array(y, int(file_sr), sr)
    except Exception as soundfile_error:
        try:
            decoded = source if not isinstance(source, (bytes, bytearray)) else io.BytesIO(bytes(source))
            return _decode_av(decoded, sr)
        except Exception as decode_error:
            raise ValueError(f"Could not read that audio ({decode_error}).") from soundfile_error


def load_audio(source: str | Path | bytes, sr: int = SAMPLE_RATE, filename: str | None = None) -> np.ndarray:
    """Load a file or a byte string as mono float32 at ``sr``.

    ``filename`` is accepted so callers can pass an upload name. The bytes
    themselves decide the format.
    """
    del filename
    return _load_decoded(source, sr)


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
