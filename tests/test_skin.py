"""Voice skins: consent, conversion toward the owner, and the booth endpoints."""

import io

import numpy as np
import pytest
import pyworld as pw
import soundfile as sf
from fastapi.testclient import TestClient

from vocallift.skin import (
    SKIN_RATE,
    SkinSettings,
    build_skin,
    convert,
    load_skin,
    provenance,
    save_skin,
    tagged_wav_bytes,
)

VOWELS = {
    "a": (730, 1090, 2440),
    "i": (270, 2290, 3010),
    "u": (300, 870, 2240),
    "e": (530, 1840, 2480),
    "o": (570, 840, 2410),
}
HOP = 120


def _envelope(freqs, formants, tract):
    spec = np.zeros_like(freqs)
    for formant, bw, amp in zip(formants, (80, 110, 150), (1.0, 0.6, 0.3)):
        spec += amp * np.exp(-0.5 * ((freqs - formant * tract) / (bw * tract)) ** 2)
    return spec / (1.0 + (freqs / 2000.0) ** 2) + 1e-4


def speaker(sequence: str, tract: float, hz: float, seed: int = 0) -> np.ndarray:
    """A synthetic talker: vowels with a falling, wobbling pitch, short breaths between them."""
    rng = np.random.default_rng(seed)
    frame_rate = SKIN_RATE / HOP
    n_bins = pw.get_cheaptrick_fft_size(SKIN_RATE) // 2 + 1
    freqs = np.linspace(0, SKIN_RATE / 2, n_bins)
    f0, sp = [], []
    for vowel in sequence:
        n = int(0.22 * frame_rate)
        t = np.arange(n) / frame_rate
        contour = hz * np.exp(rng.normal(0, 0.08) - 0.25 * t + 0.02 * np.sin(2 * np.pi * 4 * t))
        env = _envelope(freqs, VOWELS[vowel], tract)
        level = np.ones(n)
        level[:5] = np.linspace(0.2, 1, 5)
        level[-5:] = np.linspace(1, 0.2, 5)
        f0.append(contour)
        sp.append(env[None] * (0.2 * level)[:, None])
        gap = int(0.04 * frame_rate)
        f0.append(np.zeros(gap))
        sp.append(np.full((gap, n_bins), 1e-6))
    f0 = np.concatenate(f0)
    sp = np.concatenate(sp)
    ap = np.broadcast_to(np.linspace(0.02, 0.3, n_bins), sp.shape).copy()
    y = pw.synthesize(f0, sp, ap, SKIN_RATE, HOP / SKIN_RATE * 1000.0).astype(np.float32)
    return 0.6 * y / (np.max(np.abs(y)) + 1e-9)


OWNER_TEXT = "aieouoeiauaeiouiaoeuoaiuea" * 3
SPOKEN = "aiuoe"


@pytest.fixture(scope="module")
def owner_skin():
    recordings = [speaker(OWNER_TEXT, tract=1.18, hz=220, seed=1)]
    return build_skin(recordings, sr=SKIN_RATE, owner="Test Owner", consent=True, fast=True)


def _vowel_spectra(y: np.ndarray, sequence: str) -> list[np.ndarray]:
    x = y.astype(np.float64)
    f0, t = pw.dio(x, SKIN_RATE, frame_period=HOP / SKIN_RATE * 1000.0)
    sp = pw.cheaptrick(x, pw.stonemask(x, f0, t, SKIN_RATE), t, SKIN_RATE)
    per = int(0.22 * SKIN_RATE / HOP) + int(0.04 * SKIN_RATE / HOP)
    core = int(0.22 * SKIN_RATE / HOP)
    out = []
    for index in range(len(sequence)):
        start = index * per + core // 4
        seg = np.log(sp[start : start + core // 2] + 1e-12).mean(axis=0)
        band = seg[int(len(seg) * 200 / 12000) : int(len(seg) * 4000 / 12000)]
        out.append(band - band.mean())
    return out


def test_a_skin_needs_an_owner_and_their_consent():
    take = speaker(OWNER_TEXT, 1.0, 120)
    with pytest.raises(ValueError, match="permission"):
        build_skin([take], sr=SKIN_RATE, owner="Someone", consent=False)
    with pytest.raises(ValueError, match="Name"):
        build_skin([take], sr=SKIN_RATE, owner=" ", consent=True)


def test_speaker_comes_out_in_the_owners_range_and_colour(owner_skin):
    source = speaker(SPOKEN, tract=1.0, hz=115, seed=5)
    result = convert(source, owner_skin, sr=SKIN_RATE, settings=SkinSettings(fast=True), out_sr=SKIN_RATE)
    assert result.source_hz < 140
    assert abs(np.log(result.output_hz / owner_skin.meta.median_hz)) < np.log(2) / 6  # within two semitones

    out = _vowel_spectra(result.audio, SPOKEN)
    mine = _vowel_spectra(source, SPOKEN)
    theirs = _vowel_spectra(speaker(SPOKEN, tract=1.18, hz=220, seed=6), SPOKEN)
    to_owner = np.mean([np.mean((o - t) ** 2) for o, t in zip(out, theirs)])
    to_speaker = np.mean([np.mean((o - m) ** 2) for o, m in zip(out, mine)])
    assert to_owner < to_speaker
    before = np.mean([np.mean((m - t) ** 2) for m, t in zip(mine, theirs)])
    assert to_owner < 0.6 * before

    # Each vowel is still the vowel that was said: nearest owner vowel by spectrum.
    said = [min(range(len(theirs)), key=lambda j: np.mean((o - theirs[j]) ** 2)) for o in out]
    assert said == list(range(len(SPOKEN)))


def test_amount_zero_and_keep_pitch_leave_the_speaker(owner_skin):
    source = speaker(SPOKEN, tract=1.0, hz=115, seed=5)
    kept = convert(source, owner_skin, sr=SKIN_RATE, settings=SkinSettings(amount=0.0, fast=True), out_sr=SKIN_RATE)
    assert abs(np.log(kept.output_hz / kept.source_hz)) < 0.03
    melody = convert(source, owner_skin, sr=SKIN_RATE, settings=SkinSettings(pitch="keep", shift=12, fast=True), out_sr=SKIN_RATE)
    assert abs(np.log(melody.output_hz / (2 * melody.source_hz))) < 0.03


def test_skin_roundtrip_and_tagged_output(owner_skin, tmp_path):
    path = save_skin(owner_skin, tmp_path / "owner", [speaker(OWNER_TEXT, 1.18, 220)], SKIN_RATE)
    loaded = load_skin(path)
    assert loaded.meta.owner == "Test Owner"
    assert "Test Owner" in loaded.meta.consent
    assert (path / "reference.flac").is_file()
    assert np.allclose(loaded.code, owner_skin.code, atol=0.05)
    data = tagged_wav_bytes(np.zeros(2400, dtype=np.float32), SKIN_RATE, "Test Owner")
    with sf.SoundFile(io.BytesIO(data)) as handle:
        assert handle.comment == provenance("Test Owner")


def test_booth_builds_a_skin_and_speaks_through_it(tmp_path, monkeypatch):
    monkeypatch.setenv("VOCALLIFT_VOICES", str(tmp_path))
    from vocallift.api import app

    client = TestClient(app)
    assert "Voice skins" in client.get("/").text

    def wav(y):
        buf = io.BytesIO()
        sf.write(buf, y, SKIN_RATE, format="WAV")
        return buf.getvalue()

    owner = wav(speaker(OWNER_TEXT, 1.18, 220, seed=1))
    refused = client.post("/api/skins", files=[("recordings", ("me.wav", owner, "audio/wav"))], data={"owner": "Ada", "consent": "false"})
    assert refused.status_code == 400
    assert "permission" in refused.json()["detail"]

    made = client.post(
        "/api/skins",
        files=[("recordings", ("me.wav", owner, "audio/wav"))],
        data={"owner": "Ada", "consent": "true", "terms": "credit Ada", "fast": "true"},
    )
    assert made.status_code == 200, made.text
    assert made.json()["name"] == "ada"
    assert client.get("/api/skins").json()["skins"][0]["terms"] == "credit Ada"

    spoken = client.post(
        "/api/skins/speak",
        files={"file": ("say.wav", wav(speaker(SPOKEN, 1.0, 115, seed=5)), "audio/wav")},
        data={"voice": "ada", "fast": "true"},
    )
    assert spoken.status_code == 200, spoken.text
    body = spoken.json()
    assert body["owner"] == "Ada"
    assert body["output_hz"] > 1.5 * body["source_hz"]
    import base64

    with sf.SoundFile(io.BytesIO(base64.b64decode(body["audio_base64"]))) as handle:
        assert "Synthetic voice: Ada" in handle.comment

    missing = client.post("/api/skins/speak", files={"file": ("say.wav", wav(speaker(SPOKEN, 1.0, 115)), "audio/wav")}, data={"voice": "nobody"})
    assert missing.status_code == 404
