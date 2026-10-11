"""Natural pitch correction and voice-crack repair on F0 contours.

The contour is split into three layers:

* intent — the slow melodic line (vibrato removed with a short median filter)
* vibrato — the fast, musical oscillation around that line
* jitter — unmusical wobble above ~12 Hz, which is removed

Correction moves the intent toward in-key targets (or onto a guide melody)
with a short retune glide. The singer's own vibrato stays, which is what
keeps the result from sounding robotic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from vocallift.music import hz_to_midi, midi_to_hz, nearest_scale_note


@dataclass
class Note:
    start: int
    end: int  # exclusive frame index
    center: float  # sung pitch (MIDI, median)
    target: float  # pitch we pull toward (MIDI)

    @property
    def error_cents(self) -> float:
        return (self.center - self.target) * 100.0


@dataclass
class CorrectionResult:
    f0: np.ndarray
    midi_in: np.ndarray
    midi_out: np.ndarray
    voiced: np.ndarray
    repaired: np.ndarray
    notes: list[Note] = field(default_factory=list)

    def stats(self) -> dict:
        errs = np.array([abs(n.error_cents) for n in self.notes]) if self.notes else np.zeros(1)
        return {
            "notes": len(self.notes),
            "mean_abs_error_cents": float(errs.mean()),
            "notes_off_by_25c": int((errs > 25).sum()),
            "repaired_frames": int(self.repaired.sum()),
        }


def residual_cents(result: CorrectionResult) -> float:
    """How far the corrected contour still sits from its targets, in cents."""
    errs = []
    for note in result.notes:
        seg = result.midi_out[note.start : note.end]
        voiced = result.voiced[note.start : note.end]
        seg = seg[voiced]
        if seg.size:
            errs.append(abs(float(np.median(seg)) - note.target) * 100.0)
    return float(np.mean(errs)) if errs else 0.0


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs for each run of True values."""
    m = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _interp_runs(values: np.ndarray, bad: np.ndarray, good: np.ndarray) -> np.ndarray:
    out = values.copy()
    good_idx = np.flatnonzero(good & ~bad)
    if good_idx.size < 2:
        return out
    bad_idx = np.flatnonzero(bad)
    out[bad_idx] = np.interp(bad_idx, good_idx, values[good_idx])
    return out


def _nearest_stable(midi: np.ndarray, voiced: np.ndarray, bad: np.ndarray, start: int, step: int) -> float | None:
    j = start
    for _ in range(30):
        if j < 0 or j >= len(midi):
            return None
        if voiced[j] and not bad[j]:
            return float(midi[j])
        j += step
    return None


def repair_cracks(
    f0: np.ndarray,
    loudness_db: np.ndarray,
    frame_rate: float,
    sp: np.ndarray | None = None,
    sr: int = 44100,
    max_gap_ms: float = 90.0,
    max_jump_ms: float = 85.0,
    jump_semitones: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild pitch breaks: brief octave flips, and short dropouts mid-note.

    A real interval is left alone. A break qualifies only when it is short and
    the pitch on both sides of it is the same note.
    """
    f0 = f0.astype(np.float64).copy()
    voiced = f0 > 0
    repaired = np.zeros_like(voiced)
    if int(voiced.sum()) < 5:
        return f0, repaired
    midi = hz_to_midi(f0)

    win = max(3, int(0.3 * frame_rate) | 1)
    filled = _interp_runs(midi, ~voiced, voiced)
    ref = median_filter(filled, size=win, mode="nearest")
    outlier = voiced & (np.abs(midi - ref) > jump_semitones)
    max_jump = max(1, int(max_jump_ms / 1000 * frame_rate))
    for s, e in runs(outlier):
        if e - s > max_jump:
            continue
        left = _nearest_stable(midi, voiced, outlier, s - 1, -1)
        right = _nearest_stable(midi, voiced, outlier, e, +1)
        if left is None or right is None or abs(left - right) > 2.5:
            continue
        repaired[s:e] = True
    midi = _interp_runs(midi, repaired, voiced)

    max_gap = max(1, int(max_gap_ms / 1000 * frame_rate))
    hf_ratio = None
    if sp is not None:
        n_bins = sp.shape[1]
        cut = int(4000 / (sr / 2) * (n_bins - 1))
        hf_ratio = sp[:, cut:].sum(axis=1) / (sp.sum(axis=1) + 1e-12)
    for s, e in runs(~voiced):
        if s == 0 or e >= len(f0) or e - s > max_gap:
            continue
        left, right = float(midi[s - 1]), float(midi[e])
        if abs(left - right) > 3.0:
            continue
        context = loudness_db[max(0, s - 5) : s]
        if context.size and np.median(loudness_db[s:e]) < np.median(context) - 12:
            continue
        if hf_ratio is not None and np.median(hf_ratio[s:e]) > 0.35:
            continue
        repaired[s:e] = True
        midi[s:e] = np.linspace(left, right, e - s + 2)[1:-1]

    out = np.where(voiced | repaired, midi_to_hz(midi), 0.0)
    return out, repaired


def _lowpass_segments(x: np.ndarray, voiced: np.ndarray, frame_rate: float, cutoff: float) -> np.ndarray:
    out = x.copy()
    nyquist = frame_rate / 2
    cutoff = min(cutoff, nyquist * 0.9)
    if cutoff <= 1:
        return out
    sos = butter(2, cutoff / nyquist, output="sos")
    for s, e in runs(voiced):
        if e - s < 48:
            continue
        try:
            out[s:e] = sosfiltfilt(sos, x[s:e])
        except ValueError:
            pass
    return out


def _odd_window(length: int, target: int) -> int:
    size = min(target, length if length % 2 else max(1, length - 1))
    return max(1, size)


def _value_runs(values: np.ndarray) -> list[tuple[int, int]]:
    edges = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(values)]])
    return list(zip(starts.tolist(), ends.tolist()))


def _absorb_short(pieces: list[list], min_len: int) -> list[list]:
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for i, (a, b, _) in enumerate(pieces):
            if b - a >= min_len:
                continue
            if i == 0:
                j = 1
            elif i == len(pieces) - 1:
                j = i - 1
            else:
                prev_len = pieces[i - 1][1] - pieces[i - 1][0]
                next_len = pieces[i + 1][1] - pieces[i + 1][0]
                j = i - 1 if prev_len >= next_len else i + 1
            lo, hi = min(i, j), max(i, j)
            pieces[lo] = [pieces[lo][0], pieces[hi][1], pieces[j][2]]
            del pieces[hi]
            changed = True
            break
    return pieces


def _hold_positive(values: np.ndarray) -> np.ndarray:
    """Forward-fill positive pitches across zeros, then back-fill the start."""
    out = np.array(values, dtype=np.float64, copy=True)
    last = 0.0
    for i, value in enumerate(out):
        if value > 0:
            last = value
        elif last > 0:
            out[i] = last
    if out[0] <= 0:
        positive = np.flatnonzero(out > 0)
        if positive.size:
            out[: positive[0]] = out[positive[0]]
    return out


def _guide_targets(guide_slice: np.ndarray, singer_slice: np.ndarray, pitch_classes: np.ndarray) -> np.ndarray:
    guide = np.asarray(guide_slice, dtype=np.float64)
    if np.count_nonzero(guide > 0) < 2:
        return nearest_scale_note(singer_slice, pitch_classes)
    return nearest_scale_note(_hold_positive(guide), pitch_classes)


def segment_notes(
    midi: np.ndarray,
    voiced: np.ndarray,
    frame_rate: float,
    pitch_classes: np.ndarray,
    min_note_ms: float = 90.0,
    guide_midi: np.ndarray | None = None,
) -> tuple[list[Note], np.ndarray]:
    """Split the voiced contour into notes and assign each a target pitch."""
    win = max(3, int(0.18 * frame_rate) | 1)
    intent = np.array(midi, dtype=np.float64, copy=True)
    min_len = max(2, int(min_note_ms / 1000 * frame_rate))
    notes: list[Note] = []
    for s, e in runs(voiced):
        intent[s:e] = median_filter(midi[s:e], size=_odd_window(e - s, win), mode="nearest")
        if guide_midi is not None:
            cand = _guide_targets(guide_midi[s:e], midi[s:e], pitch_classes)
        else:
            cand = nearest_scale_note(intent[s:e], pitch_classes)
        pieces = _absorb_short([[a, b, float(cand[a])] for a, b in _value_runs(np.round(cand, 4))], min_len)
        for a, b, hinted in pieces:
            start, end = s + a, s + b
            center = float(np.median(midi[start:end]))
            if guide_midi is not None:
                target = float(nearest_scale_note(np.array([hinted]), pitch_classes)[0])
            else:
                target = float(nearest_scale_note(np.array([center]), pitch_classes)[0])
            if notes and notes[-1].end == start and abs(notes[-1].target - target) < 1e-6:
                notes[-1].end = end
                notes[-1].center = float(np.median(midi[notes[-1].start : end]))
            else:
                notes.append(Note(start=start, end=end, center=center, target=target))
    return notes, intent


def correct_pitch(
    f0: np.ndarray,
    frame_rate: float,
    pitch_classes: np.ndarray,
    strength: float = 0.85,
    tightness: float = 0.4,
    retune_ms: float = 50.0,
    vibrato: float = 1.0,
    add_vibrato: float = 0.0,
    jitter_cutoff_hz: float = 12.0,
    repaired: np.ndarray | None = None,
    guide_midi: np.ndarray | None = None,
) -> CorrectionResult:
    voiced = f0 > 0
    midi_in = hz_to_midi(f0)
    if int(voiced.sum()) < 3:
        return CorrectionResult(f0.copy(), midi_in, midi_in.copy(), voiced, np.zeros_like(voiced))

    midi = _lowpass_segments(midi_in, voiced, frame_rate, jitter_cutoff_hz)
    notes, intent = segment_notes(midi, voiced, frame_rate, pitch_classes, guide_midi=guide_midi)
    vib = np.where(voiced, midi - intent, 0.0)

    correction = np.zeros_like(midi)
    for note in notes:
        sl = slice(note.start, note.end)
        offset = note.target - note.center
        drift = note.center - intent[sl]
        correction[sl] = strength * offset + tightness * strength * drift

    alpha = 1.0 if retune_ms <= 0 else 1.0 - np.exp(-1000.0 / (retune_ms * frame_rate))
    smoothed = np.zeros_like(correction)
    for s, e in runs(voiced):
        acc = correction[s]
        for i in range(s, e):
            acc += alpha * (correction[i] - acc)
            smoothed[i] = acc

    out = midi + smoothed + (vibrato - 1.0) * vib
    if add_vibrato > 0:
        out += _synthetic_vibrato(notes, vib, frame_rate, len(out)) * add_vibrato
    out = np.where(voiced, out, 0.0)
    repaired_mask = repaired if repaired is not None else np.zeros_like(voiced)
    return CorrectionResult(
        f0=np.where(voiced, midi_to_hz(out), 0.0),
        midi_in=midi_in,
        midi_out=out,
        voiced=voiced,
        repaired=repaired_mask,
        notes=notes,
    )


def _synthetic_vibrato(notes: list[Note], natural_vib: np.ndarray, frame_rate: float, n: int) -> np.ndarray:
    """Fade a vibrato in on long notes that were sung straight."""
    out = np.zeros(n)
    rate_hz, depth = 5.5, 0.32
    delay, fade = int(0.22 * frame_rate), int(0.28 * frame_rate)
    for note in notes:
        length = note.end - note.start
        if length < delay + fade:
            continue
        if float(np.std(natural_vib[note.start : note.end])) > 0.12:
            continue
        t = np.arange(length) / frame_rate
        env = np.clip((np.arange(length) - delay) / max(fade, 1), 0, 1)
        release = np.clip((length - np.arange(length)) / max(0.1 * frame_rate, 1), 0, 1)
        out[note.start : note.end] = depth * env * release * np.sin(2 * np.pi * rate_hz * t)
    return out
