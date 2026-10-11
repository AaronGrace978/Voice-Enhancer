"""The singing engine: analyze, repair, retune, resynthesize, master."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from vocallift.align import map_guide
from vocallift.analysis import analyze, synthesize
from vocallift.audio import SAMPLE_RATE, fit_length, resample
from vocallift.correction import correct_pitch, repair_cracks, residual_cents
from vocallift.harmony import layer_harmony, shifted_f0
from vocallift.mastering import master_vocal
from vocallift.music import NOTE_NAMES, SCALES, detect_key, hz_to_midi, parse_key, scale_pitch_classes

_SCALE_PRETTY = {
    "major": "major",
    "minor": "minor",
    "harmonic_minor": "harmonic minor",
    "major_pentatonic": "major pentatonic",
    "minor_pentatonic": "minor pentatonic",
    "blues": "blues",
    "dorian": "dorian",
    "mixolydian": "mixolydian",
    "chromatic": "chromatic",
}


@dataclass
class EnhanceSettings:
    key: str | None = None
    scale: str = "auto"
    strength: float = 0.85
    tightness: float = 0.4
    retune_ms: float = 50.0
    vibrato: float = 1.0
    add_vibrato: float = 0.0
    repair: bool = True
    presence: float = 0.65
    reverb: float = 0.14
    harmony: bool = False
    harmony_level: float = 0.32
    fast: bool = False
    neural: bool = False
    checkpoint: str | None = None


@dataclass
class EnhanceResult:
    audio: np.ndarray
    sr: int
    hop: int
    key_root: int
    scale: str
    key_label: str
    key_source: str
    detected_label: str
    detected_mode: str
    detected_confidence: float
    correction: object
    before_cents: float
    after_cents: float
    neural_used: bool
    warning: str | None = None

    def summary(self) -> dict:
        stats = self.correction.stats()
        return {
            "key": self.key_label,
            "key_source": self.key_source,
            "detected_key": self.detected_label,
            "detected_confidence": round(self.detected_confidence, 3),
            "before_cents": round(self.before_cents, 1),
            "after_cents": round(self.after_cents, 1),
            "notes": stats["notes"],
            "repaired_frames": stats["repaired_frames"],
            "neural": self.neural_used,
            "warning": self.warning,
        }


def default_checkpoint() -> str | None:
    env = os.environ.get("VOCALLIFT_CHECKPOINT")
    if env and os.path.isfile(env):
        return env
    candidates = [Path.cwd() / "checkpoints" / "vocallift.pt"]
    if getattr(sys, "frozen", False):
        candidates.insert(0, Path(sys.executable).resolve().parent / "checkpoints" / "vocallift.pt")
    candidates.append(Path(__file__).resolve().parent.parent / "checkpoints" / "vocallift.pt")
    for path in candidates:
        if path.is_file():
            return str(path)
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, value)))


def _normalized(settings: EnhanceSettings | None) -> EnhanceSettings:
    s = settings or EnhanceSettings()
    key = s.key.strip() if isinstance(s.key, str) else None
    if not key or key.lower() == "auto":
        key = None
    scale = (s.scale or "auto").strip().lower().replace(" ", "_").replace("-", "_")
    if scale != "auto" and scale not in SCALES:
        known = ", ".join(["auto", *SCALES])
        raise ValueError(f"Unknown scale '{s.scale}'. Choose one of: {known}.")
    if key:
        parse_key(key)
    return replace(
        s,
        key=key,
        scale=scale,
        strength=_clamp(s.strength, 0.0, 1.0),
        tightness=_clamp(s.tightness, 0.0, 1.0),
        retune_ms=_clamp(s.retune_ms, 0.0, 250.0),
        vibrato=_clamp(s.vibrato, 0.0, 2.0),
        add_vibrato=_clamp(s.add_vibrato, 0.0, 1.5),
        presence=_clamp(s.presence, 0.0, 1.5),
        reverb=_clamp(s.reverb, 0.0, 0.6),
        harmony_level=_clamp(s.harmony_level, 0.0, 0.8),
    )


def _label(root: int, scale: str) -> str:
    return f"{NOTE_NAMES[root % 12]} {_SCALE_PRETTY.get(scale, scale)}"


def _prepare(y: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("That recording is empty.")
    if sr != SAMPLE_RATE:
        y = resample(y, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise ValueError("That recording is silent.")
    if peak > 1.0:
        y = y / peak
    if len(y) > 45 * sr:
        raise ValueError("Keep takes to 45 seconds or less.")
    return y, sr


def _voiced_mask(voiced: np.ndarray, n: int, hop: int, sr: int) -> np.ndarray:
    mask = np.zeros(n, dtype=np.float32)
    for index, is_voiced in enumerate(voiced):
        if not is_voiced:
            continue
        start = index * hop
        if start >= n:
            break
        mask[start : min(n, start + hop)] = 1.0
    fade = max(1, int(0.012 * sr))
    return uniform_filter1d(mask, size=fade, mode="nearest")


def _match_level(resynth: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    voiced = mask > 0.5
    if int(voiced.sum()) < 16:
        return resynth
    src = float(np.sqrt(np.mean(resynth[voiced] ** 2) + 1e-12))
    ref = float(np.sqrt(np.mean(original[voiced] ** 2) + 1e-12))
    return resynth * (ref / src)


def _fade(y: np.ndarray, sr: int) -> np.ndarray:
    fade = int(0.008 * sr)
    if len(y) <= 2 * fade:
        return y
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    y = y.copy()
    y[:fade] *= ramp
    y[-fade:] *= ramp[::-1]
    return y


def enhance(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    settings: EnhanceSettings | None = None,
    guide: np.ndarray | None = None,
) -> EnhanceResult:
    settings = _normalized(settings)
    y, sr = _prepare(y, sr)
    if guide is not None:
        guide, _ = _prepare(guide, sr)

    analysis = analyze(y, sr, fast=settings.fast)
    if settings.repair:
        f0, repaired = repair_cracks(analysis.f0, analysis.loudness_db, analysis.frame_rate, analysis.sp, sr)
    else:
        f0, repaired = analysis.f0, np.zeros_like(analysis.f0, dtype=bool)

    midi = hz_to_midi(f0)
    voiced = f0 > 0
    weights = np.clip(analysis.loudness_db[: len(f0)] + 55.0, 0.0, None)
    det_root, det_mode, det_conf = detect_key(midi, voiced, weights)
    root = parse_key(settings.key) if settings.key else det_root
    source = "given" if settings.key else "detected"
    scale = det_mode if settings.scale == "auto" else settings.scale
    classes = scale_pitch_classes(root, scale)

    guide_midi = None
    if guide is not None:
        guide_analysis = analyze(guide, sr, fast=settings.fast)
        guide_midi = map_guide(f0, guide_analysis.f0, analysis.frame_rate, classes)
        if len(guide_midi) != len(f0):
            guide_midi = np.interp(np.linspace(0, 1, len(f0)), np.linspace(0, 1, len(guide_midi)), guide_midi)

    corrected = correct_pitch(
        f0,
        analysis.frame_rate,
        classes,
        strength=settings.strength,
        tightness=settings.tightness,
        retune_ms=settings.retune_ms,
        vibrato=settings.vibrato,
        add_vibrato=settings.add_vibrato,
        repaired=repaired,
        guide_midi=guide_midi,
    )

    warning = None
    neural_used = False
    resynth = None
    if settings.neural:
        checkpoint = settings.checkpoint or default_checkpoint()
        if not checkpoint:
            warning = "Neural resynthesis is off until a Colab checkpoint is in checkpoints/vocallift.pt."
        else:
            try:
                from vocallift.neural.infer import resynthesize

                resynth = resynthesize(y, sr, corrected.f0, analysis.hop, checkpoint)
                neural_used = True
            except Exception as exc:
                warning = f"Neural resynthesis failed ({exc}). Used the signal engine instead."
                resynth = None
    if resynth is None:
        resynth = synthesize(corrected.f0, analysis.sp, analysis.ap, sr, analysis.hop)
    resynth = fit_length(resynth, len(y))

    if settings.harmony:
        harmony_f0 = shifted_f0(corrected.f0, classes, steps=2)
        harmony = synthesize(harmony_f0, analysis.sp, analysis.ap, sr, analysis.hop)
        harmony = fit_length(harmony, len(y))
        mask = _voiced_mask(corrected.voiced, len(y), analysis.hop, sr)
        resynth = layer_harmony(resynth, harmony * mask, sr, settings.harmony_level)

    mask = _voiced_mask(corrected.voiced, len(y), analysis.hop, sr)
    resynth = _match_level(resynth, y, mask)
    mixed = resynth * mask + y * (1.0 - mask)
    mastered = master_vocal(mixed, sr, presence=settings.presence, reverb_amount=settings.reverb)
    audio = _fade(mastered, sr)

    before = float(np.mean([abs(n.error_cents) for n in corrected.notes])) if corrected.notes else 0.0
    return EnhanceResult(
        audio=audio,
        sr=sr,
        hop=analysis.hop,
        key_root=root,
        scale=scale,
        key_label=_label(root, scale),
        key_source=source,
        detected_label=_label(det_root, det_mode),
        detected_mode=det_mode,
        detected_confidence=det_conf,
        correction=corrected,
        before_cents=before,
        after_cents=residual_cents(corrected),
        neural_used=neural_used,
        warning=warning,
    )
