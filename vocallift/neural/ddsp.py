"""Differentiable harmonic-plus-noise synthesizer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def upsample(x: torch.Tensor, hop: int) -> torch.Tensor:
    """(B, T, C) at frame rate -> (B, T * hop, C) at audio rate."""
    x = x.transpose(1, 2)
    x = F.interpolate(x, size=x.shape[-1] * hop, mode="linear", align_corners=False)
    return x.transpose(1, 2)


def harmonic_synth(f0: torch.Tensor, amplitudes: torch.Tensor, sr: int, hop: int) -> torch.Tensor:
    """f0: (B, T) Hz. amplitudes: (B, T, K) linear. Returns (B, T * hop)."""
    harmonics = torch.arange(1, amplitudes.shape[-1] + 1, device=f0.device, dtype=f0.dtype)
    audible = (f0[..., None] * harmonics < sr / 2 - 200).to(amplitudes.dtype)
    amplitudes = amplitudes * audible
    f0_up = upsample(f0[..., None], hop)[..., 0]
    amp_up = upsample(amplitudes, hop)
    cycles = torch.cumsum(f0_up.double() / sr, dim=1)
    phase = (cycles - torch.floor(cycles)).float() * (2 * math.pi)
    return (torch.sin(phase[..., None] * harmonics) * amp_up).sum(-1)


def filtered_noise(magnitudes: torch.Tensor, hop: int, noise: torch.Tensor | None = None) -> torch.Tensor:
    """magnitudes: (B, T, N) linear frequency response. Returns (B, T * hop)."""
    batch, frames, _bins = magnitudes.shape
    impulse = torch.fft.irfft(torch.complex(magnitudes, torch.zeros_like(magnitudes)), dim=-1)
    impulse = torch.roll(impulse, impulse.shape[-1] // 2, dims=-1)
    window = torch.hann_window(impulse.shape[-1], device=impulse.device, dtype=impulse.dtype)
    impulse = impulse * window
    if noise is None:
        noise = torch.rand(batch, frames, hop, device=magnitudes.device, dtype=magnitudes.dtype) * 2 - 1
    else:
        noise = noise[:, :frames, :hop].to(dtype=magnitudes.dtype).contiguous()
    conv_len = hop + impulse.shape[-1] - 1
    fft_len = 1 << (conv_len - 1).bit_length()
    spectrum = torch.fft.rfft(noise, fft_len) * torch.fft.rfft(impulse, fft_len)
    convolved = torch.fft.irfft(spectrum, fft_len)[..., :conv_len].transpose(1, 2).contiguous()
    total = (frames - 1) * hop + conv_len
    folded = F.fold(convolved, output_size=(1, total), kernel_size=(1, conv_len), stride=(1, hop))
    audio = folded.reshape(batch, total)
    start = impulse.shape[-1] // 2
    return audio[:, start : start + frames * hop]
