"""Run a trained resynthesizer on a full take."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from vocallift.audio import fit_length, resample
from vocallift.neural.model import VocalRestorer, pick_device

_CACHE: dict[tuple[str, str], VocalRestorer] = {}


def load_model(path: str | Path, device: torch.device | None = None) -> VocalRestorer:
    device = device or pick_device()
    key = (str(Path(path).resolve()), str(device))
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = VocalRestorer(**checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    _CACHE[key] = model
    return model


def _resample(y: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    return resample(y, src_sr, dst_sr)


def resample_f0(f0: np.ndarray, src_sr: int, src_hop: int, n_out: int, dst_sr: int, dst_hop: int) -> np.ndarray:
    """Carry a pitch contour onto another sample rate without gliding through silences."""
    n_frames = max(2, int(np.ceil(n_out / dst_hop)))
    out = np.zeros(n_frames, dtype=np.float64)
    voiced = np.asarray(f0) > 0
    if int(voiced.sum()) == 0:
        return out
    t_in = np.arange(len(f0)) * src_hop / src_sr
    t_out = (np.arange(n_frames) + 0.5) * dst_hop / dst_sr
    src = np.flatnonzero(voiced)
    out = np.interp(t_out, t_in[src], np.asarray(f0, dtype=np.float64)[src])
    nearest = np.clip(np.round(t_out / (src_hop / src_sr)).astype(int), 0, len(f0) - 1)
    out[np.asarray(f0)[nearest] <= 0] = 0.0
    return out


def _match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ry = float(np.sqrt(np.mean(y**2) + 1e-12))
    rr = float(np.sqrt(np.mean(ref**2) + 1e-12))
    return (y * (rr / ry)).astype(np.float32)


def _run(model: VocalRestorer, audio: np.ndarray, f0: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    wav = torch.tensor(audio, dtype=torch.float32, device=device)[None, :]
    pitches = torch.tensor(f0, dtype=torch.float32, device=device)[None, :]
    with torch.inference_mode():
        synth, _ = model(wav, pitches)
    out = synth[0].detach().cpu().numpy().astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)]


def _overlap(model: VocalRestorer, audio: np.ndarray, f0: np.ndarray) -> np.ndarray:
    sr, hop = model.sr, model.hop
    chunk = 8 * sr
    if len(audio) <= chunk:
        return _run(model, audio, f0)
    fade = int(0.25 * sr)
    step = chunk - fade
    window = np.ones(chunk, dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    window[:fade] = ramp
    window[-fade:] = ramp[::-1]
    mix = np.zeros(len(audio) + chunk, dtype=np.float32)
    weight = np.zeros_like(mix)
    for start in range(0, len(audio), step):
        piece = audio[start : start + chunk]
        valid = len(piece)
        if valid < chunk:
            piece = np.pad(piece, (0, chunk - valid))
        f0_start = int(round(start / hop))
        f0_end = f0_start + int(np.ceil(chunk / hop)) + 1
        out = _run(model, piece, f0[f0_start:f0_end])
        mix[start : start + chunk] += out * window
        weight[start : start + chunk] += window
        if start + valid >= len(audio):
            break
    weight = np.maximum(weight[: len(audio)], 1e-4)
    return (mix[: len(audio)] / weight).astype(np.float32)


def resynthesize(audio: np.ndarray, sr: int, f0: np.ndarray, hop: int, checkpoint: str | Path) -> np.ndarray:
    """Rebuild ``audio`` at ``f0`` (the corrected melody) and return it at ``sr``."""
    model = load_model(checkpoint)
    low = _resample(audio, sr, model.sr)
    pitches = resample_f0(f0, sr, hop, len(low), model.sr, model.hop)
    rebuilt = _overlap(model, low, pitches)
    rebuilt = _match_rms(rebuilt, low)
    high = _resample(rebuilt, model.sr, sr)
    return fit_length(high, len(audio))
