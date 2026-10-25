"""Voice skins: let anyone speak (or sing) in a voice its owner shared.

A skin is built from a few minutes of the owner's own recordings. The
signal engine needs no training and no download: every frame of the
incoming voice is matched to the frames of the owner's voice that say the
same sound, and the owner's own spectral envelopes and breathiness are
resynthesized with WORLD at a pitch moved into the owner's range. The
words and timing stay the speaker's. The timbre becomes the owner's.

The optional neural engine (``vocallift.skin_neural``) does the same
matching in WavLM feature space and vocodes with HiFi-GAN (kNN-VC).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyworld as pw
from scipy.ndimage import uniform_filter1d

from vocallift.audio import SAMPLE_RATE, frame_loudness_db, load_audio, resample

SKIN_RATE = 24000
SKIN_HOP = 120  # 5 ms
CODE_DIM = 60
MATCH_DIM = 12
CONTEXT = 2  # frames of context on each side, at a stride of CONTEXT_STRIDE
CONTEXT_STRIDE = 3
SILENCE_DB = -50.0
MAX_POOL_FRAMES = 48000  # four minutes of active speech at 5 ms
MAX_CONVERT_SECONDS = 300
MAX_PROFILE_SECONDS = 1800
PROFILE_VERSION = 1


@dataclass
class SkinMeta:
    owner: str
    consent: str
    created: float
    seconds: float
    files: int
    log_f0_mean: float
    log_f0_std: float
    terms: str = ""
    version: int = PROFILE_VERSION
    sample_rate: int = SKIN_RATE

    @property
    def median_hz(self) -> float:
        return float(np.exp(self.log_f0_mean))


@dataclass
class VoiceSkin:
    meta: SkinMeta
    code: np.ndarray  # (frames, CODE_DIM) coded spectral envelope
    ap: np.ndarray  # (frames, bands) coded aperiodicity
    voiced: np.ndarray  # (frames,) bool
    match_mean: np.ndarray
    match_std: np.ndarray
    path: Path | None = None
    _index: dict = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.path.name if self.path else slug(self.meta.owner)


@dataclass
class SkinSettings:
    amount: float = 1.0  # 0 = the speaker's own timbre, 1 = all owner
    pitch: str = "match"  # "match" moves the speaker into the owner's range, "keep" leaves the melody alone
    shift: float = 0.0  # extra semitones on top of the pitch mode
    k: int = 6
    detail: float = 0.5  # how much of each frame comes from matched owner frames rather than the warped speaker
    fast: bool = False


@dataclass
class SkinResult:
    audio: np.ndarray
    sr: int
    owner: str
    source_hz: float
    output_hz: float
    owner_hz: float
    match_distance: float
    engine: str
    warning: str | None = None

    def summary(self) -> dict:
        return {
            "owner": self.owner,
            "engine": self.engine,
            "source_hz": round(self.source_hz, 1),
            "output_hz": round(self.output_hz, 1),
            "owner_hz": round(self.owner_hz, 1),
            "match_distance": round(self.match_distance, 3),
            "warning": self.warning,
        }


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return value or "voice"


def provenance(owner: str) -> str:
    return f"Synthetic voice: {owner} voice skin, made with VocalLift. Not a recording of {owner}."


@dataclass
class _Frames:
    f0: np.ndarray
    code: np.ndarray
    ap: np.ndarray
    loud: np.ndarray

    @property
    def active(self) -> np.ndarray:
        return self.loud > SILENCE_DB


def _analyze(y: np.ndarray, fast: bool) -> _Frames:
    x = np.ascontiguousarray(y, dtype=np.float64)
    period = SKIN_HOP / SKIN_RATE * 1000.0
    if fast:
        f0, t = pw.dio(x, SKIN_RATE, f0_floor=60.0, f0_ceil=800.0, frame_period=period)
    else:
        f0, t = pw.harvest(x, SKIN_RATE, f0_floor=60.0, f0_ceil=800.0, frame_period=period)
    f0 = pw.stonemask(x, f0, t, SKIN_RATE)
    sp = pw.cheaptrick(x, f0, t, SKIN_RATE)
    ap = pw.d4c(x, f0, t, SKIN_RATE)
    loud = frame_loudness_db(y, SKIN_HOP, len(f0))
    f0 = np.where(loud < SILENCE_DB, 0.0, f0)
    code = pw.code_spectral_envelope(sp, SKIN_RATE, CODE_DIM)
    coded_ap = pw.code_aperiodicity(ap, SKIN_RATE)
    return _Frames(f0=f0, code=code.astype(np.float32), ap=coded_ap.astype(np.float32), loud=loud)


def _to_skin_rate(y: np.ndarray, sr: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.size == 0:
        raise ValueError("That recording is empty.")
    y = resample(y, sr, SKIN_RATE)
    peak = float(np.max(np.abs(y)))
    if peak < 1e-4:
        raise ValueError("That recording is silent.")
    return y / max(peak, 1.0)


def _context(match: np.ndarray) -> np.ndarray:
    """Stack a few neighbouring frames so the match hears the sound, not one slice of it."""
    n = len(match)
    parts = []
    for offset in range(-CONTEXT, CONTEXT + 1):
        index = np.clip(np.arange(n) + offset * CONTEXT_STRIDE, 0, n - 1)
        parts.append(match[index])
    return np.concatenate(parts, axis=1)


def _match_features(code: np.ndarray) -> np.ndarray:
    return code[:, 1 : MATCH_DIM + 1]


def _log_f0_stats(f0: np.ndarray) -> tuple[float, float]:
    voiced = f0[f0 > 0]
    if voiced.size < 20:
        raise ValueError("Not enough voiced sound to hear this voice. Record more speech.")
    log = np.log(voiced)
    lo, hi = np.percentile(log, [2, 98])
    log = log[(log >= lo) & (log <= hi)]
    return float(np.mean(log)), float(max(np.std(log), 0.05))


def build_skin(
    recordings: list[np.ndarray],
    sr: int = SAMPLE_RATE,
    owner: str = "",
    consent: bool = False,
    terms: str = "",
    fast: bool = False,
) -> VoiceSkin:
    """Build a skin from recordings of one voice. The owner has to say yes."""
    owner = owner.strip()
    if not owner:
        raise ValueError("Name the person whose voice this is.")
    if not consent:
        raise ValueError(
            "A voice skin needs the owner's permission. Confirm that these recordings are your own voice, "
            "or that the person speaking agreed to share it."
        )
    if not recordings:
        raise ValueError("Add at least one recording of the voice.")

    codes, aps, voiced, f0s = [], [], [], []
    seconds = 0.0
    for y in recordings:
        y = _to_skin_rate(y, sr)
        seconds += len(y) / SKIN_RATE
        if seconds > MAX_PROFILE_SECONDS:
            raise ValueError("Keep a voice skin under 30 minutes of audio. Four or five minutes is plenty.")
        frames = _analyze(y, fast)
        active = frames.active
        codes.append(frames.code[active])
        aps.append(frames.ap[active])
        voiced.append(frames.f0[active] > 0)
        f0s.append(frames.f0[active])

    code = np.concatenate(codes)
    ap = np.concatenate(aps)
    is_voiced = np.concatenate(voiced)
    f0 = np.concatenate(f0s)
    if len(code) < 400:
        raise ValueError("That is not enough speech to learn a voice. Record at least 30 seconds; a few minutes is better.")
    if len(code) > MAX_POOL_FRAMES:
        keep = np.linspace(0, len(code) - 1, MAX_POOL_FRAMES).round().astype(int)
        code, ap, is_voiced = code[keep], ap[keep], is_voiced[keep]

    mu, sigma = _log_f0_stats(f0)
    match = _match_features(code)
    meta = SkinMeta(
        owner=owner,
        consent=(
            f"{owner} confirmed this is their voice, or that its owner agreed to share it, "
            f"on {time.strftime('%Y-%m-%d', time.gmtime())}."
        ),
        created=time.time(),
        seconds=round(seconds, 2),
        files=len(recordings),
        log_f0_mean=mu,
        log_f0_std=sigma,
        terms=terms.strip(),
    )
    return VoiceSkin(
        meta=meta,
        code=code.astype(np.float32),
        ap=ap.astype(np.float32),
        voiced=is_voiced,
        match_mean=match.mean(axis=0),
        match_std=match.std(axis=0) + 1e-3,
    )


def save_skin(skin: VoiceSkin, directory: str | Path, recordings: list[np.ndarray] | None = None, sr: int = SAMPLE_RATE) -> Path:
    """Write the skin. ``recordings`` are kept at 16 kHz so the neural engine can use the skin later."""
    import soundfile as sf

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if recordings:
        gap = np.zeros(4000, dtype=np.float32)
        parts = []
        for y in recordings:
            parts += [resample(np.asarray(y, dtype=np.float32).reshape(-1), sr, 16000), gap]
        sf.write(str(path / "reference.flac"), np.clip(np.concatenate(parts), -1, 1), 16000, subtype="PCM_16")
        (path / "neural.pt").unlink(missing_ok=True)
    np.savez_compressed(
        path / "skin.npz",
        code=skin.code.astype(np.float16),
        ap=skin.ap.astype(np.float16),
        voiced=skin.voiced,
        match_mean=skin.match_mean,
        match_std=skin.match_std,
    )
    (path / "skin.json").write_text(json.dumps(asdict(skin.meta), indent=2) + "\n")
    skin.path = path
    return path


def load_skin(directory: str | Path) -> VoiceSkin:
    path = Path(directory)
    meta_path = path / "skin.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"No voice skin at {path}.")
    meta = SkinMeta(**json.loads(meta_path.read_text()))
    data = np.load(path / "skin.npz")
    return VoiceSkin(
        meta=meta,
        code=data["code"].astype(np.float32),
        ap=data["ap"].astype(np.float32),
        voiced=data["voiced"].astype(bool),
        match_mean=data["match_mean"],
        match_std=data["match_std"],
        path=path,
    )


def _pool(skin: VoiceSkin, voiced: bool) -> tuple[np.ndarray, np.ndarray]:
    cached = skin._index.get(voiced)
    if cached is None:
        rows = np.flatnonzero(skin.voiced == voiced)
        if rows.size < 8:
            rows = np.arange(len(skin.code))
        feats = _context((_match_features(skin.code) - skin.match_mean) / skin.match_std)[rows]
        cached = (rows, feats.astype(np.float32))
        skin._index[voiced] = cached
    return cached


def _knn(query: np.ndarray, pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, len(pool)))
    pool_sq = np.einsum("ij,ij->i", pool, pool)
    index = np.empty((len(query), k), dtype=np.int64)
    dist = np.empty((len(query), k), dtype=np.float32)
    for start in range(0, len(query), 512):
        block = query[start : start + 512]
        d = pool_sq[None, :] - 2.0 * block @ pool.T
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        index[start : start + len(block)] = part
        dist[start : start + len(block)] = np.take_along_axis(d, part, axis=1) + np.einsum("ij,ij->i", block, block)[:, None]
    return index, np.sqrt(np.maximum(dist, 0.0)) / np.sqrt(pool.shape[1])


def _map_f0(f0: np.ndarray, skin: VoiceSkin, settings: SkinSettings) -> tuple[np.ndarray, float]:
    voiced = f0 > 0
    out = np.zeros_like(f0)
    if not voiced.any():
        return out, 0.0
    mu_s, sd_s = _log_f0_stats(f0) if voiced.sum() >= 20 else (float(np.mean(np.log(f0[voiced]))), 0.1)
    log = np.log(f0[voiced])
    if settings.pitch == "keep":
        mapped = log
    else:
        target = skin.meta.log_f0_mean + (log - mu_s) * (skin.meta.log_f0_std / sd_s)
        mapped = log + settings.amount * (target - log)
    mapped = mapped + settings.shift * np.log(2.0) / 12.0
    out[voiced] = np.exp(mapped)
    return out, float(np.exp(mu_s))


def convert(
    y: np.ndarray,
    skin: VoiceSkin,
    sr: int = SAMPLE_RATE,
    settings: SkinSettings | None = None,
    out_sr: int = SAMPLE_RATE,
) -> SkinResult:
    """Say what ``y`` says, in the skin owner's voice."""
    settings = settings or SkinSettings()
    if settings.pitch not in {"match", "keep"}:
        raise ValueError("pitch is 'match' (move into the owner's range) or 'keep' (leave the melody alone).")
    amount = float(np.clip(settings.amount, 0.0, 1.0))
    settings = SkinSettings(
        amount=amount,
        pitch=settings.pitch,
        shift=float(np.clip(settings.shift, -24, 24)),
        k=int(np.clip(settings.k, 1, 16)),
        detail=float(np.clip(settings.detail, 0.0, 1.0)),
        fast=settings.fast,
    )

    x = _to_skin_rate(y, sr)
    if len(x) > MAX_CONVERT_SECONDS * SKIN_RATE:
        raise ValueError("Keep a conversion under five minutes. Split longer audio into parts.")
    frames = _analyze(x, settings.fast)
    active = frames.active
    if active.sum() < 10:
        raise ValueError("There is no speech in that recording.")

    src_match = _match_features(frames.code)
    src_mean = src_match[active].mean(axis=0)
    src_std = src_match[active].std(axis=0) + 1e-3
    query = _context((src_match - src_mean) / src_std).astype(np.float32)

    out_code = frames.code.copy()
    out_ap = frames.ap.copy()
    distances = []
    for voiced in (True, False):
        rows = np.flatnonzero(active & ((frames.f0 > 0) == voiced))
        if rows.size == 0:
            continue
        pool_rows, pool = _pool(skin, voiced)
        index, dist = _knn(query[rows], pool, settings.k)
        chosen = pool_rows[index]
        out_code[rows] = skin.code[chosen].mean(axis=1)
        out_ap[rows] = skin.ap[chosen].mean(axis=1)
        distances.append(dist.mean(axis=1))

    fft_size = pw.get_cheaptrick_fft_size(SKIN_RATE)
    warped, _ = _warp_toward(frames.code, active, skin, fft_size)
    out_code = uniform_filter1d(out_code, size=3, axis=0)
    out_code[~active] = warped[~active]
    out_code = settings.detail * out_code + (1.0 - settings.detail) * warped
    out_code[:, 0] = frames.code[:, 0]
    out_code = (1.0 - amount) * frames.code + amount * out_code
    out_ap = np.clip((1.0 - amount) * frames.ap + amount * out_ap, -60.0, 0.0)

    f0, source_hz = _map_f0(frames.f0, skin, settings)
    sp = pw.decode_spectral_envelope(np.ascontiguousarray(out_code, dtype=np.float64), SKIN_RATE, fft_size)
    ap = pw.decode_aperiodicity(np.ascontiguousarray(out_ap, dtype=np.float64), SKIN_RATE, fft_size)
    audio = pw.synthesize(np.ascontiguousarray(f0, dtype=np.float64), sp, ap, SKIN_RATE, SKIN_HOP / SKIN_RATE * 1000.0)
    audio = _level(audio.astype(np.float32), x)
    audio = resample(audio[: len(x)], SKIN_RATE, out_sr)

    voiced_out = f0[f0 > 0]
    return SkinResult(
        audio=audio,
        sr=out_sr,
        owner=skin.meta.owner,
        source_hz=source_hz,
        output_hz=float(np.exp(np.mean(np.log(voiced_out)))) if voiced_out.size else 0.0,
        owner_hz=skin.meta.median_hz,
        match_distance=float(np.mean(np.concatenate(distances))) if distances else 0.0,
        engine="signal",
    )


def _log_envelope(code: np.ndarray, fft_size: int) -> np.ndarray:
    sp = pw.decode_spectral_envelope(np.ascontiguousarray(code, dtype=np.float64), SKIN_RATE, fft_size)
    return np.log(sp + 1e-12)


def _warp_toward(code: np.ndarray, active: np.ndarray, skin: VoiceSkin, fft_size: int) -> tuple[np.ndarray, float]:
    """Stretch the speaker's vocal tract toward the owner's, then lend it the owner's average colour.

    Every sound keeps its own shape, so nothing is lost from the words; only
    the length of the tract and the long-term tilt change.
    """
    log_sp = _log_envelope(code, fft_size)
    owner_mean = _log_envelope(skin.code.mean(axis=0, keepdims=True), fft_size)[0]
    source_mean = log_sp[active].mean(axis=0)
    bins = np.arange(log_sp.shape[1], dtype=np.float64)
    band = slice(int(log_sp.shape[1] * 150 / (SKIN_RATE / 2)), int(log_sp.shape[1] * 5000 / (SKIN_RATE / 2)))

    def warp(spec: np.ndarray, alpha: float) -> np.ndarray:
        position = np.clip(bins / alpha, 0, len(bins) - 1)
        lo = np.floor(position).astype(int)
        hi = np.minimum(lo + 1, len(bins) - 1)
        frac = position - lo
        return spec[..., lo] * (1 - frac) + spec[..., hi] * frac

    best_alpha, best = 1.0, np.inf
    for alpha in np.arange(0.80, 1.26, 0.01):
        shaped = warp(source_mean, alpha)
        diff = shaped[band] - owner_mean[band]
        score = float(np.mean((diff - diff.mean()) ** 2))
        if score < best:
            best_alpha, best = float(alpha), score
    shifted = warp(log_sp, best_alpha)
    colour = owner_mean - warp(source_mean, best_alpha)
    smooth = uniform_filter1d(colour, size=max(3, len(bins) // 24))
    out = pw.code_spectral_envelope(np.ascontiguousarray(np.exp(shifted + smooth)), SKIN_RATE, CODE_DIM)
    out[:, 0] = code[:, 0]
    return out.astype(np.float32), best_alpha


def _level(audio: np.ndarray, reference: np.ndarray) -> np.ndarray:
    n = min(len(audio), len(reference))
    ref = float(np.sqrt(np.mean(reference[:n] ** 2) + 1e-12))
    got = float(np.sqrt(np.mean(audio[:n] ** 2) + 1e-12))
    out = audio * (ref / got)
    peak = float(np.max(np.abs(out))) + 1e-9
    if peak > 0.98:
        out *= 0.98 / peak
    return out


def speak_as(
    y: np.ndarray,
    skin: VoiceSkin,
    sr: int = SAMPLE_RATE,
    settings: SkinSettings | None = None,
    engine: str = "signal",
    out_sr: int = SAMPLE_RATE,
) -> SkinResult:
    """Convert with the chosen engine. If the neural engine cannot run, say so and use the signal engine."""
    settings = settings or SkinSettings()
    if engine not in {"signal", "neural"}:
        raise ValueError("engine is 'signal' or 'neural'.")
    if engine == "neural":
        from vocallift import skin_neural

        ok, reason = skin_neural.available()
        if ok and skin.path is not None:
            try:
                x = _to_skin_rate(y, sr)
                if len(x) > MAX_CONVERT_SECONDS * SKIN_RATE:
                    raise ValueError("Keep a conversion under five minutes. Split longer audio into parts.")
                neural = skin_neural.NeuralSkin.for_skin(skin.path)
                audio = neural.convert(x, SKIN_RATE, out_sr, k=max(4, settings.k), amount=settings.amount)
                return SkinResult(
                    audio=audio,
                    sr=out_sr,
                    owner=skin.meta.owner,
                    source_hz=_mean_hz(x, SKIN_RATE),
                    output_hz=_mean_hz(audio, out_sr),
                    owner_hz=skin.meta.median_hz,
                    match_distance=0.0,
                    engine="neural",
                )
            except ValueError:
                raise
            except Exception as exc:
                reason = f"The neural engine failed ({exc})."
        elif ok:
            reason = "Save the skin before using the neural engine."
        result = convert(y, skin, sr, settings, out_sr)
        result.warning = f"{reason} Used the signal engine instead."
        return result
    return convert(y, skin, sr, settings, out_sr)


def _mean_hz(y: np.ndarray, sr: int) -> float:
    x = np.ascontiguousarray(y, dtype=np.float64)
    f0, t = pw.dio(x, sr, f0_floor=60.0, f0_ceil=800.0, frame_period=10.0)
    f0 = pw.stonemask(x, f0, t, sr)
    voiced = f0[f0 > 0]
    return float(np.exp(np.mean(np.log(voiced)))) if voiced.size else 0.0


def voices_dir() -> Path:
    env = os.environ.get("VOCALLIFT_VOICES")
    if env:
        return Path(env)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "voices"
    return Path.cwd() / "voices"


def list_skins(root: str | Path | None = None) -> list[VoiceSkin]:
    base = Path(root) if root else voices_dir()
    if not base.is_dir():
        return []
    skins = []
    for path in sorted(base.iterdir()):
        if (path / "skin.json").is_file():
            try:
                skins.append(load_skin(path))
            except Exception:
                continue
    return skins


def find_skin(name: str, root: str | Path | None = None) -> VoiceSkin:
    direct = Path(name)
    if (direct / "skin.json").is_file():
        return load_skin(direct)
    base = Path(root) if root else voices_dir()
    path = base / slug(name)
    if not (path / "skin.json").is_file():
        known = ", ".join(s.name for s in list_skins(base)) or "none yet"
        raise FileNotFoundError(f"No voice skin called '{name}'. Skins in {base}: {known}.")
    return load_skin(path)


def save_tagged(path: str | Path, audio: np.ndarray, sr: int, owner: str) -> None:
    """Write a WAV that says, in its own metadata, that it is a synthetic voice."""
    Path(path).write_bytes(tagged_wav_bytes(audio, sr, owner))


def tagged_wav_bytes(audio: np.ndarray, sr: int, owner: str) -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    with sf.SoundFile(buf, "w", samplerate=sr, channels=1, subtype="PCM_16", format="WAV") as handle:
        handle.comment = provenance(owner)
        handle.software = "VocalLift voice skin"
        handle.write(np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0))
    return buf.getvalue()


def load_recordings(paths: list[str | Path]) -> list[np.ndarray]:
    return [load_audio(p) for p in paths]
