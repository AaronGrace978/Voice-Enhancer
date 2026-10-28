"""The neural voice-skin engine: kNN-VC (WavLM features, HiFi-GAN vocoder).

kNN-VC (Baas, van Niekerk, Kamper, Interspeech 2023, MIT licence) is the
same idea as the signal engine with better ears and a better mouth: each
20 ms of the speaker is described by WavLM, replaced by the average of the
owner's closest WavLM frames, and vocoded back to audio. It needs no
training on the owner, only their recordings, and sounds most like them
with five minutes or more.

The model code ships in ``vocallift/vendor/knnvc``. The weights (about
1.3 GB) are downloaded once, on first use or from the booth's download
button, and kept in the torch hub cache.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path

import numpy as np

from vocallift.audio import resample

NEURAL_RATE = 16000
CHUNK_SECONDS = 20
FEATURE_LAYER = 6
RELEASE = "https://github.com/bshall/knn-vc/releases/download/v0.1/"
WEIGHTS = {
    "WavLM-Large.pt": 1261965425,
    "prematch_g_02500000.pt": 66214643,
}

HIFIGAN_CONFIG = {
    "resblock": "1",
    "upsample_rates": [10, 8, 2, 2],
    "upsample_kernel_sizes": [20, 16, 4, 4],
    "upsample_initial_channel": 512,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "hubert_dim": 1024,
    "hifi_dim": 512,
}

_MODEL = None
_MODEL_LOCK = threading.Lock()
_DOWNLOAD = {"active": False, "done": 0, "total": sum(WEIGHTS.values()), "error": None}
_DOWNLOAD_LOCK = threading.Lock()


def available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "The neural engine needs PyTorch: pip install torch."
    return True, ""


def weights_dir() -> Path:
    env = os.environ.get("VOCALLIFT_NEURAL_WEIGHTS")
    if env:
        return Path(env)
    try:
        import torch

        return Path(torch.hub.get_dir()) / "checkpoints"
    except ImportError:
        return Path.home() / ".cache" / "torch" / "hub" / "checkpoints"


def weights_ready() -> bool:
    folder = weights_dir()
    return all((folder / name).is_file() and (folder / name).stat().st_size == size for name, size in WEIGHTS.items())


def download_status() -> dict:
    with _DOWNLOAD_LOCK:
        status = dict(_DOWNLOAD)
    ok, reason = available()
    status.update(torch=ok, reason=reason, ready=weights_ready(), folder=str(weights_dir()))
    return status


def download_weights(progress=None) -> None:
    """Fetch the kNN-VC weights into the torch hub cache, resuming nothing and verifying sizes."""
    folder = weights_dir()
    folder.mkdir(parents=True, exist_ok=True)
    done = sum(size for name, size in WEIGHTS.items() if (folder / name).is_file() and (folder / name).stat().st_size == size)
    for name, size in WEIGHTS.items():
        target = folder / name
        if target.is_file() and target.stat().st_size == size:
            continue
        partial = target.with_suffix(".part")
        with urllib.request.urlopen(RELEASE + name, timeout=60) as response, open(partial, "wb") as handle:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
                done += len(block)
                if progress:
                    progress(done)
        if partial.stat().st_size != size:
            partial.unlink(missing_ok=True)
            raise OSError(f"{name} downloaded incompletely. Try again.")
        partial.replace(target)


def start_download() -> dict:
    """Download the weights in the background, so the booth can show progress."""
    with _DOWNLOAD_LOCK:
        if _DOWNLOAD["active"] or weights_ready():
            return download_status()
        _DOWNLOAD.update(active=True, done=0, error=None)

    def progress(done: int) -> None:
        with _DOWNLOAD_LOCK:
            _DOWNLOAD["done"] = done

    def run() -> None:
        try:
            download_weights(progress)
        except Exception as exc:
            with _DOWNLOAD_LOCK:
                _DOWNLOAD["error"] = str(exc)
        finally:
            with _DOWNLOAD_LOCK:
                _DOWNLOAD["active"] = False

    threading.Thread(target=run, daemon=True).start()
    return download_status()


class _KnnVC:
    def __init__(self, wavlm, hifigan, device) -> None:
        self.wavlm = wavlm
        self.hifigan = hifigan
        self.device = device


def load_model() -> _KnnVC:
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        import torch

        from vocallift.vendor.knnvc.hifigan.models import Generator
        from vocallift.vendor.knnvc.hifigan.utils import AttrDict
        from vocallift.vendor.knnvc.wavlm.WavLM import WavLM, WavLMConfig

        if not weights_ready():
            download_weights()
        folder = weights_dir()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(folder / "WavLM-Large.pt", map_location="cpu", weights_only=True)
        wavlm = WavLM(WavLMConfig(checkpoint["cfg"]))
        wavlm.load_state_dict(checkpoint["model"])
        wavlm = wavlm.to(device).eval()

        hifigan = Generator(AttrDict(HIFIGAN_CONFIG))
        state = torch.load(folder / "prematch_g_02500000.pt", map_location="cpu", weights_only=True)
        hifigan.load_state_dict(state["generator"])
        hifigan.remove_weight_norm()
        hifigan = hifigan.to(device).eval()
        _MODEL = _KnnVC(wavlm, hifigan, device)
        return _MODEL


def selftest() -> dict:
    """Run the WavLM encoder and HiFi-GAN vocoder with random weights. Proves PyTorch and the model code work, no download."""
    import torch

    from vocallift.vendor.knnvc.hifigan.models import Generator
    from vocallift.vendor.knnvc.hifigan.utils import AttrDict
    from vocallift.vendor.knnvc.wavlm.WavLM import WavLM, WavLMConfig

    torch.manual_seed(0)
    small = {"encoder_layers": FEATURE_LAYER, "encoder_embed_dim": 1024, "encoder_ffn_embed_dim": 256, "encoder_attention_heads": 4}
    wavlm = WavLM(WavLMConfig(small)).eval()
    hifigan = Generator(AttrDict(HIFIGAN_CONFIG))
    hifigan.remove_weight_norm()
    hifigan.eval()
    model = _KnnVC(wavlm, hifigan, torch.device("cpu"))
    t = np.arange(NEURAL_RATE) / NEURAL_RATE
    feats = _features(model, (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32))
    with torch.inference_mode():
        audio = hifigan(feats[None]).squeeze().numpy()
    if feats.shape[1] != 1024 or not np.isfinite(audio).all():
        raise RuntimeError("The neural engine produced invalid output.")
    return {"torch": torch.__version__, "frames": int(feats.shape[0]), "samples": int(audio.shape[-1])}


def _features(model: _KnnVC, y: np.ndarray):
    import torch

    feats = []
    step = CHUNK_SECONDS * NEURAL_RATE
    with torch.inference_mode():
        for start in range(0, len(y), step):
            piece = y[start : start + step]
            if len(piece) < NEURAL_RATE // 4:
                continue
            x = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32))[None].to(model.device)
            out = model.wavlm.extract_features(x, output_layer=FEATURE_LAYER, ret_layer_results=False)[0]
            feats.append(out.squeeze(0).cpu())
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
        feats = [_features(model, resample(y, sr, NEURAL_RATE)) for y in recordings]
        return cls(torch.cat(feats, dim=0))

    @classmethod
    def for_skin(cls, directory: str | Path) -> "NeuralSkin":
        """Load the cached WavLM matching set for a skin, building it from the skin's reference audio once."""
        import soundfile as sf
        import torch

        path = Path(directory)
        cache = path / "neural.pt"
        if cache.is_file():
            return cls(torch.load(cache, map_location="cpu", weights_only=True).float())
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
        query = _features(model, x)
        amount = float(np.clip(amount, 0.0, 1.0))
        k = int(np.clip(k, 1, 16))
        with torch.inference_mode():
            pool = self.matching_set.to(model.device).float()
            query = query.to(model.device)
            unit_pool = pool / pool.norm(dim=-1, keepdim=True)
            best = []
            for start in range(0, len(query), 1024):
                block = query[start : start + 1024]
                block = block / block.norm(dim=-1, keepdim=True)
                best.append((block @ unit_pool.T).topk(k=k, dim=-1).indices)
            matched = pool[torch.cat(best)].mean(dim=1)
            feats = (1.0 - amount) * query + amount * matched
            out = model.hifigan(feats[None]).squeeze().cpu().numpy().astype(np.float32)
        ref = float(np.sqrt(np.mean(x**2) + 1e-12))
        out *= ref / float(np.sqrt(np.mean(out**2) + 1e-12))
        peak = float(np.max(np.abs(out))) + 1e-9
        if peak > 0.98:
            out *= 0.98 / peak
        return resample(out, NEURAL_RATE, out_sr)
