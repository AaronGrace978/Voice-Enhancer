"""The differentiable resynthesizer learns to undo a damaged phrase."""

import numpy as np
import torch

from vocallift.neural.ddsp import filtered_noise, harmonic_synth
from vocallift.neural.train import save_checkpoint, spectral_loss, train_steps


def test_harmonic_synth_passes_gradients():
    f0 = torch.full((2, 10), 196.0)
    amplitudes = torch.zeros(2, 10, 6)
    amplitudes[..., 0] = 0.25
    amplitudes.requires_grad_(True)
    audio = harmonic_synth(f0, amplitudes, 16000, 160)
    assert audio.shape == (2, 1600)
    audio.pow(2).mean().backward()
    assert torch.isfinite(amplitudes.grad).all()
    assert amplitudes.grad.abs().sum() > 0

    magnitudes = torch.full((1, 6, 16), 0.05, requires_grad=True)
    noise = filtered_noise(magnitudes, 160)
    assert noise.shape[-1] == 6 * 160
    noise.pow(2).mean().backward()
    assert magnitudes.grad.abs().sum() > 0


def test_pretraining_lowers_the_loss(tmp_path):
    model, losses = train_steps(steps=8, batch=2, seconds=0.55, seed=2, log_every=0)
    assert all(np.isfinite(losses))
    assert np.mean(losses[-3:]) < np.mean(losses[:3])
    path = tmp_path / "vocallift.pt"
    save_checkpoint(model, path)

    from vocallift.neural.infer import load_model, resynthesize

    loaded = load_model(path, device=torch.device("cpu"))
    f0 = np.zeros(40, dtype=np.float64)
    f0[8:32] = 220
    audio = np.zeros(40 * 160, dtype=np.float32)
    audio[8 * 160 : 32 * 160] = 0.2
    rebuilt = resynthesize(audio, loaded.sr, f0, loaded.hop, path)
    assert rebuilt.shape == audio.shape
    assert np.isfinite(rebuilt).all()
    silent = torch.zeros(1, 1600)
    assert torch.isfinite(spectral_loss(silent, silent))
