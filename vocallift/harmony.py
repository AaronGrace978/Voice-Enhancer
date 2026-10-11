"""A diatonic double tracked a few milliseconds late."""

from __future__ import annotations

import numpy as np

from vocallift.music import diatonic_shift, hz_to_midi, midi_to_hz


def shifted_f0(f0: np.ndarray, pitch_classes: np.ndarray, steps: int = 2) -> np.ndarray:
    """Move every voiced frame by ``steps`` scale degrees. 2 is a third."""
    out = np.zeros_like(f0, dtype=np.float64)
    voiced = f0 > 0
    if not np.any(voiced):
        return out
    midi = np.round(hz_to_midi(f0[voiced]))
    out[voiced] = midi_to_hz(diatonic_shift(midi, pitch_classes, steps))
    return out


def layer_harmony(lead: np.ndarray, harmony: np.ndarray, sr: int, level: float = 0.32, delay_ms: float = 16.0) -> np.ndarray:
    delay = int(delay_ms * sr / 1000.0)
    doubled = np.pad(np.asarray(harmony, dtype=np.float32), (delay, 0))[: len(lead)]
    if len(doubled) < len(lead):
        doubled = np.pad(doubled, (0, len(lead) - len(doubled)))
    return np.asarray(lead, dtype=np.float32) + np.float32(level) * doubled
