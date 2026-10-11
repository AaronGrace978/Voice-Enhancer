"""Place a guide melody onto the singer's timing.

When the singer outlines the same number of notes as the guide, each sung
note inherits the guide's pitch — timing can be loose and the pitches still
land. Otherwise the guide is time-stretched across the performance, which is
what pulls a monotone or wandering take onto the tune.
"""

from __future__ import annotations

import numpy as np

from vocallift.correction import segment_notes
from vocallift.music import hz_to_midi, nearest_scale_note


def map_guide(
    user_f0: np.ndarray,
    guide_f0: np.ndarray,
    frame_rate: float,
    pitch_classes: np.ndarray,
) -> np.ndarray:
    """Return guide MIDI (0 where the singer is unvoiced) on the singer's frames."""
    user_f0 = np.asarray(user_f0, dtype=np.float64)
    guide_f0 = np.asarray(guide_f0, dtype=np.float64)
    user_midi = hz_to_midi(user_f0)
    guide_midi = hz_to_midi(guide_f0)
    user_voiced = user_f0 > 0
    guide_voiced = guide_f0 > 0
    mapped = np.zeros(len(user_f0), dtype=np.float64)
    if int(guide_voiced.sum()) < 2 or int(user_voiced.sum()) < 2:
        return mapped

    user_notes, _ = segment_notes(user_midi, user_voiced, frame_rate, pitch_classes)
    guide_notes, _ = segment_notes(guide_midi, guide_voiced, frame_rate, pitch_classes)
    if user_notes and len(user_notes) == len(guide_notes):
        for sung, guide in zip(user_notes, guide_notes):
            target = float(nearest_scale_note(np.array([guide.center]), pitch_classes)[0])
            mapped[sung.start : sung.end] = target
        return mapped

    src = np.flatnonzero(guide_voiced)
    stretched = np.interp(
        np.linspace(0.0, 1.0, len(user_f0)),
        np.linspace(0.0, 1.0, len(guide_f0))[src],
        guide_midi[src],
    )
    mapped[user_voiced] = stretched[user_voiced]
    return mapped
