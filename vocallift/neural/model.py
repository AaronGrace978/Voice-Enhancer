"""A small differentiable singing synthesizer.

The DSP engine decides the pitch. This network reads the singer's tone from
a mel spectrogram and rebuilds the voice at the pitch it is given, so a
corrected melody can be resynthesized instead of only shifted.

Train it with ``vocallift.neural.train``. An untrained network is not loaded
by the booth.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from vocallift.neural.ddsp import filtered_noise, harmonic_synth


def default_config() -> dict:
    return {
        "n_mels": 64,
        "n_harmonics": 32,
        "n_noise": 24,
        "hidden": 96,
        "hop": 160,
        "sr": 16000,
        "n_fft": 1024,
    }


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class VocalRestorer(nn.Module):
    def __init__(
        self,
        n_mels: int = 64,
        n_harmonics: int = 32,
        n_noise: int = 24,
        hidden: int = 96,
        hop: int = 160,
        sr: int = 16000,
        n_fft: int = 1024,
    ):
        super().__init__()
        self.config = {
            "n_mels": n_mels,
            "n_harmonics": n_harmonics,
            "n_noise": n_noise,
            "hidden": hidden,
            "hop": hop,
            "sr": sr,
            "n_fft": n_fft,
        }
        self.hop = hop
        self.sr = sr
        self.n_harmonics = n_harmonics
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=40.0,
            f_max=min(7600.0, sr / 2 - 200),
            center=True,
            power=2.0,
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.harm = nn.Linear(hidden * 2, n_harmonics)
        self.noise = nn.Linear(hidden * 2, n_noise)
        self.amp = nn.Linear(hidden * 2, 1)
        self.f0_residual = nn.Linear(hidden * 2, 1)
        self.register_buffer("noise_bank", torch.rand(1, 4000, hop) * 2 - 1, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.constant_(self.noise.bias, -3.0)

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        logmel = torch.log(self.mel(audio) + 1e-5)
        hidden = self.conv(logmel).transpose(1, 2)
        hidden, _ = self.gru(hidden)
        return hidden

    def _noise(self, frames: int, batch: int) -> torch.Tensor:
        if not self.training:
            return torch.rand(batch, frames, self.hop, device=self.noise_bank.device) * 2 - 1
        bank = self.noise_bank
        if frames > bank.shape[1]:
            repeats = (frames + bank.shape[1] - 1) // bank.shape[1]
            bank = bank.repeat(1, repeats, 1)
        return bank[:, :frames, :].expand(batch, -1, -1)

    def forward(self, audio: torch.Tensor, f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """audio (B, samples), f0 (B, frames) in Hz. Returns audio and the f0 used."""
        hidden = self.encode(audio)
        frames = hidden.shape[1]
        if f0.shape[1] != frames:
            f0 = F.interpolate(f0[:, None, :], size=frames, mode="linear", align_corners=False)[:, 0, :]
        residual = 0.5 * torch.tanh(self.f0_residual(hidden).squeeze(-1))
        f0_out = torch.where(f0 > 0, f0 * torch.pow(2.0, residual / 12.0), torch.zeros_like(f0))
        distribution = F.softplus(self.harm(hidden))
        distribution = distribution / distribution.sum(-1, keepdim=True).clamp(min=1e-5)
        amplitude = F.softplus(self.amp(hidden))
        noise_mag = 1e-3 * F.softplus(self.noise(hidden))
        voiced = (f0 > 0).to(amplitude.dtype).unsqueeze(-1)
        harmonic = harmonic_synth(f0_out, amplitude * distribution * voiced, self.sr, self.hop)
        noise = filtered_noise(noise_mag * voiced, self.hop, noise=self._noise(frames, audio.shape[0]))
        return harmonic + noise, f0_out
