"""Self-supervised training for the singing resynthesizer.

Each step builds a clean phrase and a damaged twin: the twin is shifted,
cracked, noised, and dropped out, but it keeps the same vowel. The network
hears the twin and is told the clean pitch, and it has to sing the clean
phrase back. That is the behavior the booth wants — new pitch, same tone.

Fine-tuning on a person's own takes is ordinary reconstruction: analyze
their pitch, resynthesize them, and also do it from a noisy copy.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vocallift.neural.ddsp import harmonic_synth
from vocallift.neural.model import VocalRestorer, default_config, pick_device


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :n]
    target = target[..., :n]
    if n < 256:
        return F.l1_loss(pred, target)
    loss = pred.new_zeros(())
    used = 0
    for n_fft, hop in ((512, 128), (1024, 256), (2048, 512)):
        if n < n_fft:
            continue
        window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
        pred_mag = torch.stft(pred, n_fft, hop, n_fft, window, return_complex=True).abs()
        target_mag = torch.stft(target, n_fft, hop, n_fft, window, return_complex=True).abs()
        loss = loss + F.l1_loss(torch.log(pred_mag + 1e-5), torch.log(target_mag + 1e-5))
        loss = loss + 0.1 * F.l1_loss(pred_mag, target_mag)
        used += 1
    return loss / max(used, 1)


def _vowel() -> list[tuple[float, float, float]]:
    return [
        (float(np.random.uniform(280, 800)), 85.0, 1.0),
        (float(np.random.uniform(900, 2300)), 120.0, 0.65),
        (float(np.random.uniform(2300, 3400)), 160.0, 0.3),
    ]


def _harmonics(f0: torch.Tensor, formants: list[tuple[float, float, float]], n_harmonics: int) -> torch.Tensor:
    ks = torch.arange(1, n_harmonics + 1, device=f0.device, dtype=f0.dtype)
    freqs = f0[:, None] * ks[None, :]
    envelope = torch.zeros_like(freqs)
    for freq, bw, amp in formants:
        envelope = envelope + amp * torch.exp(-0.5 * ((freqs - freq) / bw) ** 2)
    envelope = envelope / ks
    return envelope / envelope.sum(-1, keepdim=True).clamp(min=1e-6)


def random_batch(batch: int, seconds: float, config: dict, device: torch.device):
    """Return degraded audio, clean audio, and the clean f0 the model must follow."""
    sr, hop, n_harmonics = config["sr"], config["hop"], config["n_harmonics"]
    frames = max(8, int(seconds * sr / hop))
    frame_rate = sr / hop
    f0 = torch.zeros(batch, frames, device=device)
    f0_src = torch.zeros(batch, frames, device=device)
    amp = torch.zeros(batch, frames, device=device)
    harm = torch.zeros(batch, frames, n_harmonics, device=device)
    harm_src = torch.zeros(batch, frames, n_harmonics, device=device)
    for b in range(batch):
        formants = _vowel()
        shift = float(np.random.uniform(-2.0, 2.0))
        pos = 0
        while pos < frames:
            dur = max(4, int(np.random.uniform(0.18, 0.42) * frame_rate))
            gap = int(np.random.uniform(0.04, 0.08) * frame_rate)
            midi = float(np.random.choice([57, 60, 62, 64, 65, 67, 69, 71, 72]))
            end = min(frames, pos + dur)
            length = end - pos
            t = torch.arange(length, device=device) / frame_rate
            rate = float(np.random.uniform(4.8, 6.1))
            contour = midi + 0.18 * torch.sin(2 * math.pi * rate * t)
            hz = 440.0 * torch.pow(2.0, (contour - 69.0) / 12.0)
            env = torch.ones(length, device=device)
            edge = max(1, min(6, length // 3))
            env[:edge] = torch.linspace(0.15, 1.0, edge, device=device)
            env[-edge:] = torch.linspace(1.0, 0.15, edge, device=device)
            f0[b, pos:end] = hz
            amp[b, pos:end] = 0.45 * env
            harm[b, pos:end] = _harmonics(hz, formants, n_harmonics)
            src = hz * (2.0 ** (shift / 12.0))
            if length > 14 and np.random.rand() < 0.75:
                src = src.clone()
                c0 = length // 2
                src[c0 : c0 + 4] *= 2.0
            f0_src[b, pos:end] = src
            harm_src[b, pos:end] = _harmonics(src, formants, n_harmonics)
            pos = end + gap
    with torch.no_grad():
        clean = harmonic_synth(f0, amp.unsqueeze(-1) * harm, sr, hop)
        source = harmonic_synth(f0_src, amp.unsqueeze(-1) * harm_src, sr, hop)
    rms = source.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-4)
    snr = torch.empty(batch, 1, device=device).uniform_(22.0, 34.0)
    degraded = source + rms / (10 ** (snr / 20.0)) * torch.randn_like(source)
    for b in range(batch):
        width = int(np.random.uniform(0.02, 0.05) * sr)
        if degraded.shape[-1] > width + 1:
            start = int(np.random.randint(0, degraded.shape[-1] - width))
            degraded[b, start : start + width] = 0
    peak = clean.abs().amax(dim=-1, keepdim=True).clamp(min=1e-3)
    return (degraded / peak * 0.7).detach(), (clean / peak * 0.7).detach(), f0.detach()


def _f0_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    frames = min(pred.shape[-1], target.shape[-1])
    pred = pred[:, :frames]
    target = target[:, :frames]
    voiced = target > 0
    if int(voiced.sum()) == 0:
        return pred.sum() * 0
    return F.l1_loss(pred[voiced], target[voiced])


def train_steps(
    steps: int = 200,
    batch: int = 4,
    seconds: float = 1.2,
    lr: float = 1e-3,
    device: torch.device | None = None,
    model: VocalRestorer | None = None,
    seed: int = 0,
    log_every: int = 20,
) -> tuple[VocalRestorer, list[float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or pick_device()
    config = default_config() if model is None else model.config
    model = model or VocalRestorer(**config)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    for step in range(steps):
        degraded, clean, f0 = random_batch(batch, seconds, model.config, device)
        pred, f0_hat = model(degraded, f0)
        loss = spectral_loss(pred, clean) + 0.02 * _f0_loss(f0_hat, f0)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became {loss.item()} at step {step}.")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if log_every and step % log_every == 0:
            print(f"step {step:4d}  loss {losses[-1]:.4f}", flush=True)
    return model, losses


def save_checkpoint(model: VocalRestorer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "config": model.config, "model": model.state_dict()}, path)


def _track_f0(y: np.ndarray, sr: int, hop: int) -> np.ndarray:
    import pyworld as pw

    period = hop / sr * 1000.0
    x = np.ascontiguousarray(y, dtype=np.float64)
    f0, times = pw.dio(x, sr, f0_floor=65.0, f0_ceil=900.0, frame_period=period)
    return pw.stonemask(x, f0, times, sr).astype(np.float32)


def _excerpts(paths: list[str], sr: int, hop: int, clip_s: float) -> list[tuple[np.ndarray, np.ndarray]]:
    from vocallift.audio import load_audio

    clips = []
    n = int(clip_s * sr)
    for path in paths:
        y = load_audio(path, sr=sr)
        f0 = _track_f0(y, sr, hop)
        if len(y) <= n:
            clips.append((y, f0))
            continue
        for start in range(0, len(y) - n + 1, max(1, n // 2)):
            a, b = start, start + n
            clips.append((y[a:b], f0[int(a / hop) : int(b / hop)]))
    return clips


def finetune_files(
    model: VocalRestorer,
    paths: list[str],
    steps: int = 80,
    clip_s: float = 3.0,
    lr: float = 5e-4,
    device: torch.device | None = None,
) -> list[float]:
    """Adapt the resynthesizer to recordings of one voice."""
    if not paths:
        raise ValueError("Give at least one vocal recording to fine-tune on.")
    device = device or pick_device()
    model.to(device).train()
    clips = _excerpts(paths, model.sr, model.hop, clip_s)
    if not clips:
        raise ValueError("Those recordings did not contain any audio.")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        audio, f0 = clips[step % len(clips)]
        wav = torch.tensor(audio, dtype=torch.float32, device=device)[None, :]
        f0_t = torch.tensor(f0, dtype=torch.float32, device=device)[None, :]
        if f0_t.shape[-1] < 2:
            continue
        pred, _ = model(wav, f0_t)
        noisy = wav + 0.005 * torch.randn_like(wav)
        pred_noisy, _ = model(noisy, f0_t)
        loss = spectral_loss(pred, wav) + 0.5 * spectral_loss(pred_noisy, wav)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 10 == 0:
            print(f"finetune {step:4d}  loss {losses[-1]:.4f}", flush=True)
    return losses
