"""A quiet chord bed so a dry vocal can be previewed in a key."""

from __future__ import annotations

import numpy as np

from vocallift.audio import peak_normalize
from vocallift.music import midi_to_hz

# Intervals above the root, one chord per tuple.
_MAJOR = ((0, 4, 7), (7, 11, 14), (9, 12, 16), (5, 9, 12))  # I V vi IV
_MINOR = ((0, 3, 7), (8, 12, 15), (3, 7, 10), (10, 14, 17))  # i VI III VII
_MINOR_SCALES = {"minor", "harmonic_minor", "minor_pentatonic", "dorian", "blues"}


def family(scale: str, detected_mode: str) -> str:
    if scale in _MINOR_SCALES:
        return "minor"
    if scale in {"major", "major_pentatonic", "mixolydian"}:
        return "major"
    return "minor" if detected_mode == "minor" else "major"


def _voice(base: int, intervals: tuple[int, ...]) -> list[int]:
    notes = []
    for interval in intervals:
        note = base + interval
        while note > base + 14:
            note -= 12
        while note < base - 3:
            note += 12
        notes.append(note)
    return notes


def render_backing(n_samples: int, sr: int, root_pc: int, mode: str = "major", bpm: float = 96.0) -> np.ndarray:
    """Loop a four-chord pad of length ``n_samples``."""
    progression = _MINOR if mode == "minor" else _MAJOR
    base = 48 + (root_pc % 12)
    chord_len = max(1, int(60.0 / bpm * 2.0 * sr))
    y = np.zeros(n_samples, dtype=np.float64)
    attack = int(0.03 * sr)
    release = int(0.06 * sr)
    for index in range(int(np.ceil(n_samples / chord_len))):
        start = index * chord_len
        end = min(n_samples, start + chord_len)
        length = end - start
        t = np.arange(length) / sr
        env = np.ones(length)
        a = min(length // 2, attack)
        r = min(length // 2, release)
        if a:
            env[:a] = np.linspace(0.0, 1.0, a)
        if r:
            env[-r:] *= np.linspace(1.0, 0.0, r)
        chord = np.zeros(length)
        for note in _voice(base, progression[index % len(progression)]):
            hz = float(midi_to_hz(np.array([note]))[0])
            chord += np.sin(2 * np.pi * hz * t)
            chord += 0.35 * np.sin(2 * np.pi * hz * 2 * t)
            chord += 0.5 * np.sin(2 * np.pi * (hz / 2) * t)
        y[start:end] += chord * env
    return peak_normalize(y.astype(np.float32), peak_db=-14.0)


def mix_with_backing(
    vocal: np.ndarray,
    sr: int,
    root_pc: int,
    scale: str,
    detected_mode: str,
) -> np.ndarray:
    backing = render_backing(len(vocal), sr, root_pc, family(scale, detected_mode))
    mix = vocal.astype(np.float32) + 0.9 * backing
    return peak_normalize(mix, peak_db=-1.5)
