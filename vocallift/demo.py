"""Synthetic takes used by the booth demo and the tests.

These are sung by the WORLD vocoder: a real pitch contour with vibrato,
an octave crack, and a short dropout, rendered through a vowel spectrum.
They are not recordings of a person, and the booth says so.
"""

from __future__ import annotations

import numpy as np
import pyworld as pw

from vocallift.audio import HOP_LENGTH, SAMPLE_RATE
from vocallift.music import midi_to_hz

# A C-major phrase: do mi so do la fa mi re do.
MELODY = [60, 64, 67, 72, 69, 65, 64, 62, 60]


def _vowel(freqs: np.ndarray, f1: float = 650, f2: float = 1400, f3: float = 2600) -> np.ndarray:
    spec = np.zeros_like(freqs)
    for freq, bw, amp in ((f1, 90, 1.0), (f2, 120, 0.65), (f3, 160, 0.3)):
        spec += amp * np.exp(-0.5 * ((freqs - freq) / bw) ** 2)
    spec = spec / (1.0 + (freqs / 1800.0) ** 2) + 1e-4
    return spec


def render_melody(
    midis: list[int],
    cents,
    sr: int = SAMPLE_RATE,
    vibrato_cents: float = 25.0,
    octave_flip: int | None = None,
    dropout: int | None = None,
    note_s: float = 0.34,
    gap_s: float = 0.07,
    seed: int = 1,
) -> np.ndarray:
    """Render ``midis`` with a per-note cents offset. Scalar ``cents`` applies to every note."""
    if np.isscalar(cents):
        offsets = [float(cents)] * len(midis)
    else:
        offsets = [float(c) for c in cents]
    hop = HOP_LENGTH
    frame_rate = sr / hop
    note_n = int(note_s * frame_rate)
    gap_n = int(gap_s * frame_rate)
    pitches: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    for index, midi in enumerate(midis):
        t = np.arange(note_n) / frame_rate
        contour = midi + offsets[index] / 100.0 + (vibrato_cents / 100.0) * np.sin(2 * np.pi * 5.2 * t)
        if octave_flip == index:
            c0 = note_n // 2
            contour[c0 : c0 + 4] += 12
        hz = midi_to_hz(contour)
        env = np.ones(note_n)
        edge = min(6, note_n // 4)
        env[:edge] = np.linspace(0.2, 1.0, edge)
        env[-edge:] = np.linspace(1.0, 0.15, edge)
        if dropout == index:
            c0 = note_n // 2
            hz[c0 : c0 + 4] = 0
            env[c0 : c0 + 4] = 1.0
        pitches.append(hz)
        energy.append(env)
        if index != len(midis) - 1:
            pitches.append(np.zeros(gap_n))
            energy.append(np.full(gap_n, 0.02))
    f0 = np.concatenate(pitches)
    env = np.concatenate(energy)
    n_bins = pw.get_cheaptrick_fft_size(sr) // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    sp = _vowel(freqs)[None, :] * (0.2 * env)[:, None]
    ap = np.broadcast_to(np.linspace(0.02, 0.22, n_bins), sp.shape).copy()
    y = pw.synthesize(f0.astype(np.float64), sp.astype(np.float64), ap, sr, hop / sr * 1000.0)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) + 1e-6
    y *= np.float32(0.7 / peak)
    rng = np.random.default_rng(seed)
    y = y + rng.normal(0, 0.002, len(y)).astype(np.float32)
    return y


def offkey_take(sr: int = SAMPLE_RATE) -> np.ndarray:
    """The right melody, sung sour: every note is off, one cracks, one drops out.

    Offsets stay inside half a semitone. Past that, the nearest scale tone can
    be the neighbor, and the way back to the melody is a guide track.
    """
    cents = [-42, 35, -38, 44, -28, 40, -45, 33, -36]
    return render_melody(MELODY, cents, sr=sr, octave_flip=2, dropout=5, seed=2)


def guide_take(sr: int = SAMPLE_RATE) -> np.ndarray:
    """The same phrase, sung in tune. Used as the melody to follow."""
    return render_melody(MELODY, 0, sr=sr, vibrato_cents=18, seed=3)


def flat_take(sr: int = SAMPLE_RATE) -> np.ndarray:
    """About a minor third flat: nearest-note correction picks the wrong degrees."""
    return render_melody(MELODY, -140, sr=sr, vibrato_cents=28, octave_flip=4, seed=4)
