"""Music theory helpers: scales, key detection, note names."""

from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES: dict[str, list[int]] = {
    "chromatic": list(range(12)),
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "major_pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def hz_to_midi(f0: np.ndarray) -> np.ndarray:
    f0 = np.asarray(f0, dtype=np.float64)
    out = np.zeros_like(f0)
    mask = f0 > 0
    out[mask] = 69.0 + 12.0 * np.log2(f0[mask] / 440.0)
    return out


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    midi = np.asarray(midi, dtype=np.float64)
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def parse_key(key: str) -> int:
    key = key.strip().upper().replace("♯", "#").replace("♭", "B")
    flats = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
    key = flats.get(key, key)
    if key not in NOTE_NAMES:
        raise ValueError(f"Unknown key: {key}")
    return NOTE_NAMES.index(key)


def scale_pitch_classes(root: int, scale: str) -> np.ndarray:
    if scale not in SCALES:
        raise ValueError(f"Unknown scale '{scale}'. Choose from: {', '.join(SCALES)}")
    return np.array(sorted((root + s) % 12 for s in SCALES[scale]))


def detect_key(midi: np.ndarray, voiced: np.ndarray, weights: np.ndarray | None = None) -> tuple[int, str, float]:
    """Estimate (root, 'major'|'minor', confidence) from a sung pitch contour."""
    m = midi[voiced]
    if m.size < 10:
        return 0, "major", 0.0
    w = np.ones_like(m) if weights is None else weights[voiced]
    hist = np.zeros(12)
    np.add.at(hist, np.round(m).astype(int) % 12, w)
    best = (0, "major", -np.inf)
    scores = []
    for root in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            r = np.corrcoef(hist, np.roll(profile, root))[0, 1]
            r = -1.0 if np.isnan(r) else r
            scores.append(r)
            if r > best[2]:
                best = (root, mode, r)
    scores = np.sort(scores)
    confidence = float(np.clip((scores[-1] - scores[-2]) * 5, 0, 1))
    return best[0], best[1], confidence


def nearest_scale_note(midi: np.ndarray, pitch_classes: np.ndarray) -> np.ndarray:
    """Snap each (fractional) MIDI value to the closest note within ``pitch_classes``."""
    midi = np.asarray(midi, dtype=np.float64)
    candidates = []
    base = np.floor(midi / 12.0) * 12.0
    for octave in (-12.0, 0.0, 12.0):
        for pc in pitch_classes:
            candidates.append(base + octave + pc)
    cand = np.stack(candidates, axis=-1)
    idx = np.argmin(np.abs(cand - midi[..., None]), axis=-1)
    return np.take_along_axis(cand, idx[..., None], axis=-1)[..., 0]


def diatonic_shift(midi_notes: np.ndarray, pitch_classes: np.ndarray, steps: int) -> np.ndarray:
    """Move in-scale notes by ``steps`` scale degrees (e.g. 2 = a diatonic third)."""
    pcs = list(pitch_classes)
    out = np.array(midi_notes, dtype=np.float64)
    for i, n in enumerate(out):
        r = int(round(n))
        if r % 12 not in pcs:
            r = int(nearest_scale_note(np.array([r]), pitch_classes)[0])
        octave, pc = divmod(r, 12)
        deg = pcs.index(pc) + steps
        o, d = divmod(deg, len(pcs))
        out[i] = (octave + o) * 12 + pcs[d]
    return out
