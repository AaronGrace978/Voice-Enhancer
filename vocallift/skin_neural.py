"""The neural voice-skin engine: kNN-VC (WavLM features, HiFi-GAN vocoder).

kNN-VC (Baas, van Niekerk, Kamper, Interspeech 2023, MIT licence) is the
same idea as the signal engine with better ears and a better mouth: each
20 ms of the speaker is described by WavLM, replaced by the average of the
owner's closest WavLM frames, and vocoded back to audio. It needs no
training on the owner, only their recordings, and sounds most like them
with five minutes or more.

The first use downloads about 1.3 GB of weights through torch.hub.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vocallift.audio import resample

NEURAL_RATE = 16000
CHUNK_SECONDS = 20
_MODEL = None


def available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
        import torchaudio  # noqa: F401
    except ImportError:
        return False, "The neural engine needs PyTorch: pip install torch torchaudio."
    return True, ""


def load_model():
    global _MODEL
    if _MODEL is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=device)
    return _MODEL


def _features(model, y: np.ndarray, vad: bool):
    import torch

    feats = []
    step = CHUNK_SECONDS * NEURAL_RATE
    for start in range(0, len(y), step):
        piece = y[start : start + step]
        if len(piece) < NEURAL_RATE // 4:
            continue
        x = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))[None]
        feats.append(model.get_features(x, vad_trigger_level=7 if vad else 0).cpu())
    if not feats:
        raise ValueError("That recording is too short for the neural engine.")
    return torch.cat(feats, dim=0)


class NeuralSkin:
    def __init__(self, matching_set) -> None:
        self.matching_set = matching_set

    @classmethod
    def from_recordings(cls, recordings: list[np.ndarray], sr: int) -> "NeuralSkin":
        import torch

        model = load_model()
        feats = [_features(model, resample(y, sr, NEURAL_RATE), vad=True) for y in recordings]
        return cls(torch.cat(feats, dim=0))

    @classmethod
    def for_skin(cls, directory: str | Path) -> "NeuralSkin":
        """Load the cached WavLM matching set for a skin, building it from the skin's reference audio once."""
        import soundfile as sf
        import torch

        path = Path(directory)
        cache = path / "neural.pt"
        if cache.is_file():
            return cls(torch.load(cache, map_location="cpu").float())
        reference = path / "reference.flac"
        if not reference.is_file():
            raise FileNotFoundError("This skin has no reference audio, so the neural engine cannot use it. Rebuild it.")
        y, sr = sf.read(str(reference), dtype="float32")
        skin = cls.from_recordings([y], sr)
        torch.save(skin.matching_set.half(), cache)
        return skin

    def convert(self, y: np.ndarray, sr: int, out_sr: int, k: int = 4, amount: float = 1.0) -> np.ndarray:
        import torch

        model = load_model()
        x = resample(y, sr, NEURAL_RATE)
        query = _features(model, x, vad=False)
        amount = float(np.clip(amount, 0.0, 1.0))
        with torch.inference_mode():
            pool = self.matching_set.to(model.device).float()
            query = query.to(model.device)
            unit_pool = pool / pool.norm(dim=-1, keepdim=True)
            best = []
            for start in range(0, len(query), 1024):
                block = query[start : start + 1024]
                block = block / block.norm(dim=-1, keepdim=True)
                best.append((block @ unit_pool.T).topk(k=int(np.clip(k, 1, 16)), dim=-1).indices)
            matched = pool[torch.cat(best)].mean(dim=1)
            feats = (1.0 - amount) * query + amount * matched
            out = model.vocode(feats[None]).cpu().squeeze().numpy().astype(np.float32)
        ref = float(np.sqrt(np.mean(x**2) + 1e-12))
        out *= ref / float(np.sqrt(np.mean(out**2) + 1e-12))
        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 0.98:
            out *= 0.98 / peak
        return resample(out, NEURAL_RATE, out_sr)
