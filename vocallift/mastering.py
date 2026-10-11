"""Lightweight vocal mastering chain: EQ, de-esser, compressor, reverb, limiter."""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, lfilter


def _biquad(kind: str, f0: float, sr: int, q: float = 0.707, gain_db: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """RBJ audio-EQ-cookbook biquad coefficients."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    cos, sin = np.cos(w0), np.sin(w0)
    alpha = sin / (2 * q)
    if kind == "highpass":
        b = [(1 + cos) / 2, -(1 + cos), (1 + cos) / 2]
        den = [1 + alpha, -2 * cos, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * a, -2 * cos, 1 - alpha * a]
        den = [1 + alpha / a, -2 * cos, 1 - alpha / a]
    elif kind in ("lowshelf", "highshelf"):
        sq = 2 * np.sqrt(a) * alpha
        sign = 1 if kind == "lowshelf" else -1
        b = [
            a * ((a + 1) - sign * (a - 1) * cos + sq),
            sign * 2 * a * ((a - 1) - sign * (a + 1) * cos),
            a * ((a + 1) - sign * (a - 1) * cos - sq),
        ]
        den = [
            (a + 1) + sign * (a - 1) * cos + sq,
            -sign * 2 * ((a - 1) + sign * (a + 1) * cos),
            (a + 1) + sign * (a - 1) * cos - sq,
        ]
    elif kind == "bandpass":
        b = [alpha, 0.0, -alpha]
        den = [1 + alpha, -2 * cos, 1 - alpha]
    else:
        raise ValueError(kind)
    b, den = np.array(b), np.array(den)
    return b / den[0], den / den[0]


def _apply(y: np.ndarray, coeffs: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return lfilter(coeffs[0], coeffs[1], y)


def _envelope(x: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """Peak envelope follower, vectorised by block for speed."""
    a_att = np.exp(-1.0 / (attack_ms * 1e-3 * sr))
    a_rel = np.exp(-1.0 / (release_ms * 1e-3 * sr))
    block = 64
    n = int(np.ceil(len(x) / block))
    peaks = np.abs(np.pad(x, (0, n * block - len(x)))).reshape(n, block).max(axis=1)
    env = np.zeros(n)
    att_b, rel_b = a_att**block, a_rel**block
    level = 0.0
    for i, p in enumerate(peaks):
        coef = att_b if p > level else rel_b
        level = coef * level + (1 - coef) * p
        env[i] = level
    return np.repeat(env, block)[: len(x)]


def compress(y: np.ndarray, sr: int, threshold_db: float = -20, ratio: float = 3.0, attack_ms: float = 5, release_ms: float = 120) -> np.ndarray:
    env_db = 20 * np.log10(_envelope(y, sr, attack_ms, release_ms) + 1e-9)
    over = np.maximum(env_db - threshold_db, 0)
    gain_db = -over * (1 - 1 / ratio)
    out = y * 10 ** (gain_db / 20)
    return out


def de_ess(y: np.ndarray, sr: int, freq: float = 6500, threshold_db: float = -30, max_cut_db: float = 8) -> np.ndarray:
    band = _apply(y, _biquad("bandpass", freq, sr, q=1.2))
    env_db = 20 * np.log10(_envelope(band, sr, 1, 60) + 1e-9)
    cut_db = np.clip(env_db - threshold_db, 0, max_cut_db)
    gain = 10 ** (-cut_db / 20)
    return y - band * (1 - gain)


def reverb(y: np.ndarray, sr: int, amount: float = 0.2, decay_s: float = 1.6, seed: int = 7) -> np.ndarray:
    if amount <= 0:
        return y
    rng = np.random.default_rng(seed)
    n = int(decay_s * sr)
    t = np.arange(n) / sr
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / decay_s)
    pre = int(0.02 * sr)
    ir = np.concatenate([np.zeros(pre), ir])
    ir = _apply(ir, _biquad("highshelf", 4000, sr, gain_db=-9))
    ir /= np.sqrt(np.sum(ir**2)) + 1e-9
    wet = fftconvolve(y, ir)[: len(y)]
    wet *= np.sqrt(np.mean(y**2) + 1e-12) / (np.sqrt(np.mean(wet**2)) + 1e-12)
    return (1 - amount * 0.5) * y + amount * wet


def limit(y: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20)
    return np.tanh(y / ceiling) * ceiling


def master_vocal(y: np.ndarray, sr: int, presence: float = 0.5, reverb_amount: float = 0.15, target_rms_db: float = -18.0) -> np.ndarray:
    x = y.astype(np.float64)
    x = _apply(x, _biquad("highpass", 85, sr, q=0.707))
    x = _apply(x, _biquad("peak", 320, sr, q=1.0, gain_db=-2.5 * presence))
    x = _apply(x, _biquad("peak", 3800, sr, q=0.9, gain_db=3.5 * presence))
    x = _apply(x, _biquad("highshelf", 10000, sr, gain_db=3.0 * presence))
    x = de_ess(x, sr)
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    x *= 10 ** (target_rms_db / 20) / rms
    x = compress(x, sr, threshold_db=target_rms_db + 2, ratio=3.0)
    x = reverb(x, sr, amount=reverb_amount)
    rms = np.sqrt(np.mean(x**2) + 1e-12)
    x *= 10 ** (target_rms_db / 20) / rms
    return limit(x).astype(np.float32)
