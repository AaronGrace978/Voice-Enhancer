"""Vocal analysis built on the WORLD vocoder (pitch, spectral envelope, aperiodicity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyworld as pw

from vocallift.audio import HOP_LENGTH, SAMPLE_RATE, frame_loudness_db

F0_FLOOR = 65.0
F0_CEIL = 1100.0


@dataclass
class VocalAnalysis:
    f0: np.ndarray  # Hz per frame, 0 where unvoiced
    sp: np.ndarray  # spectral envelope (frames, fft_size // 2 + 1)
    ap: np.ndarray  # aperiodicity (frames, fft_size // 2 + 1)
    loudness_db: np.ndarray
    sr: int
    hop: int

    @property
    def voiced(self) -> np.ndarray:
        return self.f0 > 0

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    @property
    def n_frames(self) -> int:
        return len(self.f0)


def extract_f0(y: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP_LENGTH, fast: bool = False) -> np.ndarray:
    x = y.astype(np.float64)
    period = hop / sr * 1000.0
    if fast:
        f0, t = pw.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=period)
    else:
        f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=period)
    return pw.stonemask(x, f0, t, sr)


def analyze(y: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP_LENGTH, fast: bool = False) -> VocalAnalysis:
    x = y.astype(np.float64)
    period = hop / sr * 1000.0
    if fast:
        f0, t = pw.dio(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=period)
    else:
        f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, f0_ceil=F0_CEIL, frame_period=period)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    ap = pw.d4c(x, f0, t, sr)
    loud = frame_loudness_db(y, hop, len(f0))
    f0 = np.where(loud < -55.0, 0.0, f0)
    return VocalAnalysis(f0=f0, sp=sp, ap=ap, loudness_db=loud, sr=sr, hop=hop)


def synthesize(f0: np.ndarray, sp: np.ndarray, ap: np.ndarray, sr: int = SAMPLE_RATE, hop: int = HOP_LENGTH) -> np.ndarray:
    period = hop / sr * 1000.0
    y = pw.synthesize(
        np.ascontiguousarray(f0, dtype=np.float64),
        np.ascontiguousarray(sp, dtype=np.float64),
        np.ascontiguousarray(ap, dtype=np.float64),
        sr,
        period,
    )
    return y.astype(np.float32)
