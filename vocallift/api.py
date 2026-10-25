"""The vocal booth: a small local server with a browser UI."""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from vocallift import __version__
from vocallift.audio import SAMPLE_RATE, load_audio, to_wav_bytes
from vocallift.chords import mix_with_backing
from vocallift.demo import flat_take, guide_take, offkey_take
from vocallift.music import note_name
from vocallift.pipeline import EnhanceSettings, default_checkpoint, enhance

def _web_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "web"
    return Path(__file__).resolve().parent.parent / "web"


WEB = _web_dir()
app = FastAPI(title="VocalLift", version=__version__)
_DEMO_CACHE: dict[str, dict] = {}


def _b64(y: np.ndarray, sr: int) -> str:
    return base64.b64encode(to_wav_bytes(y, sr)).decode("ascii")


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "on", "yes"}


def _settings(
    key: str,
    scale: str,
    strength: str,
    tightness: str,
    retune_ms: str,
    vibrato: str,
    add_vibrato: str,
    repair: str,
    harmony: str,
    reverb: str,
    presence: str,
    neural: str,
    fast: str,
) -> EnhanceSettings:
    def num(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    return EnhanceSettings(
        key=key,
        scale=scale,
        strength=num(strength, 0.85),
        tightness=num(tightness, 0.4),
        retune_ms=num(retune_ms, 50),
        vibrato=num(vibrato, 1.0),
        add_vibrato=num(add_vibrato, 0.0),
        repair=_flag(repair, True),
        harmony=_flag(harmony),
        reverb=num(reverb, 0.14),
        presence=num(presence, 0.65),
        neural=_flag(neural),
        fast=_flag(fast),
    )


def _payload(result, original: np.ndarray | None = None, mix: np.ndarray | None = None, elapsed_ms: float = 0) -> dict:
    corr = result.correction
    hop, sr = result.hop, result.sr
    n = len(corr.midi_in)
    step = max(1, n // 700)
    times, midi_in, midi_out = [], [], []
    for i in range(0, n, step):
        times.append(round(i * hop / sr, 4))
        if corr.voiced[i]:
            midi_in.append(round(float(corr.midi_in[i]), 3))
            midi_out.append(round(float(corr.midi_out[i]), 3))
        else:
            midi_in.append(None)
            midi_out.append(None)
    notes = []
    for note in corr.notes[:100]:
        seg = corr.midi_out[note.start : note.end]
        voiced = corr.voiced[note.start : note.end]
        landed = float(np.median(seg[voiced])) if np.any(voiced) else note.target
        notes.append(
            {
                "t": round(note.start * hop / sr, 3),
                "sung": note_name(note.center),
                "target": note_name(note.target),
                "before_cents": round(note.error_cents),
                "after_cents": round((landed - note.target) * 100),
            }
        )
    body = {
        "audio_base64": _b64(result.audio, sr),
        "sample_rate": sr,
        "duration": round(len(result.audio) / sr, 3),
        "key": result.key_label,
        "key_source": result.key_source,
        "detected_key": result.detected_label,
        "detected_confidence": round(result.detected_confidence, 3),
        "before_cents": round(result.before_cents, 1),
        "after_cents": round(result.after_cents, 1),
        "repaired_frames": int(corr.repaired.sum()),
        "notes": notes,
        "times": times,
        "midi_in": midi_in,
        "midi_out": midi_out,
        "neural": result.neural_used,
        "warning": result.warning,
        "elapsed_ms": round(elapsed_ms),
    }
    if original is not None:
        body["original_base64"] = _b64(original, sr)
    if mix is not None:
        body["mix_base64"] = _b64(mix, sr)
    return body


def _run(y: np.ndarray, settings: EnhanceSettings, guide: np.ndarray | None, backing: bool, original: np.ndarray | None):
    started = time.perf_counter()
    try:
        result = enhance(y, SAMPLE_RATE, settings, guide=guide)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    mix = mix_with_backing(result.audio, result.sr, result.key_root, result.scale, result.detected_mode) if backing else None
    return _payload(result, original, mix, (time.perf_counter() - started) * 1000)


def _read(upload: UploadFile | None) -> np.ndarray | None:
    if upload is None or not upload.filename:
        return None
    data = upload.file.read()
    if not data:
        return None
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(413, "Keep uploads under 25 MB.")
    try:
        return load_audio(data, filename=upload.filename)
    except Exception as exc:
        raise HTTPException(400, f"Could not read {upload.filename}.") from exc


@app.get("/api/health")
def health() -> dict:
    from vocallift.skin_neural import available

    checkpoint = default_checkpoint()
    return {
        "ok": True,
        "version": __version__,
        "neural": bool(checkpoint),
        "checkpoint": Path(checkpoint).name if checkpoint else None,
        "skin_neural": available()[0],
    }


@app.post("/api/enhance")
def enhance_take(
    file: UploadFile = File(...),
    guide: UploadFile | None = File(None),
    key: str = Form("auto"),
    scale: str = Form("auto"),
    strength: str = Form("0.85"),
    tightness: str = Form("0.4"),
    retune_ms: str = Form("50"),
    vibrato: str = Form("1"),
    add_vibrato: str = Form("0"),
    repair: str = Form("true"),
    harmony: str = Form("false"),
    reverb: str = Form("0.14"),
    presence: str = Form("0.65"),
    neural: str = Form("false"),
    fast: str = Form("false"),
    backing: str = Form("false"),
) -> dict:
    audio = _read(file)
    if audio is None:
        raise HTTPException(400, "Choose a vocal take first.")
    return _run(audio, _settings(key, scale, strength, tightness, retune_ms, vibrato, add_vibrato, repair, harmony, reverb, presence, neural, fast), _read(guide), _flag(backing), None)


@app.post("/api/demo")
def demo(mode: str = Form("offkey"), backing: str = Form("true")) -> dict:
    """A fixed preset, so the first listen is the engine and not the slider positions."""
    mode = mode if mode in {"offkey", "guide"} else "offkey"
    cached = _DEMO_CACHE.get(mode)
    if cached is not None and _flag(backing, True):
        return cached
    settings = EnhanceSettings(key="C", scale="major", strength=1.0, tightness=0.55, retune_ms=35, vibrato=1.0, repair=True, reverb=0.16, presence=0.7)
    if mode == "guide":
        take = flat_take()
        payload = _run(take, settings, guide_take(), _flag(backing, True), take)
    else:
        take = offkey_take()
        payload = _run(take, settings, None, _flag(backing, True), take)
    payload["demo"] = mode
    if _flag(backing, True):
        _DEMO_CACHE[mode] = payload
    return payload


def _skin_card(item) -> dict:
    return {
        "name": item.name,
        "owner": item.meta.owner,
        "seconds": round(item.meta.seconds, 1),
        "hz": round(item.meta.median_hz),
        "terms": item.meta.terms,
        "consent": item.meta.consent,
    }


@app.get("/api/skins")
def skins() -> dict:
    from vocallift.skin import list_skins

    return {"skins": [_skin_card(item) for item in list_skins()]}


@app.post("/api/skins")
def build_voice_skin(
    recordings: list[UploadFile] = File(...),
    owner: str = Form(""),
    consent: str = Form("false"),
    terms: str = Form(""),
    fast: str = Form("false"),
) -> dict:
    from vocallift.skin import build_skin, save_skin, slug, voices_dir

    audio = [y for y in (_read(item) for item in recordings) if y is not None]
    started = time.perf_counter()
    try:
        built = build_skin(audio, owner=owner, consent=_flag(consent), terms=terms, fast=_flag(fast))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    save_skin(built, voices_dir() / slug(owner), audio)
    card = _skin_card(built)
    card["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
    return card


@app.post("/api/skins/speak")
def speak_in_skin(
    file: UploadFile = File(...),
    voice: str = Form(...),
    engine: str = Form("signal"),
    amount: str = Form("1"),
    pitch: str = Form("match"),
    shift: str = Form("0"),
    fast: str = Form("false"),
) -> dict:
    from vocallift.skin import SkinSettings, find_skin, provenance, speak_as, tagged_wav_bytes

    audio = _read(file)
    if audio is None:
        raise HTTPException(400, "Choose a recording to speak through the skin.")
    try:
        skin = find_skin(voice)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc

    def num(value: str, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    settings = SkinSettings(amount=num(amount, 1.0), pitch=pitch, shift=num(shift, 0.0), fast=_flag(fast))
    started = time.perf_counter()
    try:
        result = speak_as(audio, skin, SAMPLE_RATE, settings, engine=engine)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    body = result.summary()
    body.update(
        audio_base64=base64.b64encode(tagged_wav_bytes(result.audio, result.sr, result.owner)).decode("ascii"),
        sample_rate=result.sr,
        provenance=provenance(result.owner),
        terms=skin.meta.terms,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
    )
    return body


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
