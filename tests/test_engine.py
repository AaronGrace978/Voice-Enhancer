"""The singing engine: pitch repair, melody following, the booth, and training."""

import io

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from vocallift.audio import frame_loudness_db, load_audio, to_wav_bytes
from vocallift.chords import render_backing
from vocallift.correction import correct_pitch, repair_cracks, residual_cents
from vocallift.demo import MELODY, flat_take, guide_take, offkey_take
from vocallift.harmony import shifted_f0
from vocallift.mastering import master_vocal
from vocallift.music import detect_key, hz_to_midi, midi_to_hz, nearest_scale_note, parse_key, scale_pitch_classes
from vocallift.pipeline import EnhanceSettings, enhance


def _compress(values):
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def test_scale_snap_and_key():
    classes = scale_pitch_classes(parse_key("C"), "major")
    snapped = nearest_scale_note(np.array([60.4, 63.2, 66.8]), classes)
    assert list(np.round(snapped).astype(int)) == [60, 64, 67]
    midi = np.repeat(np.array([60, 64, 67, 72, 67, 64, 60], dtype=float), 24)
    root, mode, confidence = detect_key(midi, np.ones_like(midi, dtype=bool))
    assert (root, mode) == (0, "major")
    assert confidence > 0


def test_crack_is_rebuilt_and_a_real_leap_is_not():
    rate = 100.0
    midi = np.full(120, 67.0)
    midi[50:54] += 12  # a 40 ms octave flip
    f0 = midi_to_hz(midi)
    loud = np.full(len(f0), -18.0)
    fixed, mask = repair_cracks(f0, loud, rate)
    assert mask[50:54].all()
    assert abs(hz_to_midi(fixed[52]) - 67) < 0.4

    leap = np.full(120, 60.0)
    leap[40:] = 67.0
    fixed_leap, leap_mask = repair_cracks(midi_to_hz(leap), loud, rate)
    assert not leap_mask.any()
    assert abs(hz_to_midi(fixed_leap[80]) - 67) < 0.2


def test_dropout_fills_when_loud_and_stays_open_when_quiet():
    rate = 100.0
    f0 = midi_to_hz(np.full(80, 69.0))
    f0[30:34] = 0
    loud = np.full(len(f0), -16.0)
    _, mask = repair_cracks(f0, loud, rate)
    assert mask[30:34].all()
    loud[30:34] = -50
    _, quiet = repair_cracks(f0, loud, rate)
    assert not quiet[30:34].any()


def test_vibrato_survives_and_straight_tone_can_grow_some():
    rate = 100.0
    t = np.arange(400) / rate
    sung = 69 - 0.4 + 0.28 * np.sin(2 * np.pi * 5.2 * t)
    f0 = midi_to_hz(sung)
    classes = scale_pitch_classes(0, "chromatic")
    kept = correct_pitch(f0, rate, classes, strength=1, tightness=0, retune_ms=5, vibrato=1)
    removed = correct_pitch(f0, rate, classes, strength=1, tightness=0, retune_ms=5, vibrato=0)

    def depth(midi):
        from scipy.ndimage import median_filter

        core = np.asarray(midi)[40:-40]
        slow = median_filter(core, size=21)
        return float(np.std(core - slow))

    assert depth(kept.midi_out) > 0.75 * depth(kept.midi_in)
    assert depth(removed.midi_out) < 0.45 * depth(kept.midi_in)
    assert residual_cents(kept) < 12

    straight = correct_pitch(midi_to_hz(np.full(400, 68.6)), rate, classes, strength=1, add_vibrato=1, retune_ms=5)
    assert depth(straight.midi_out) > depth(straight.midi_in) + 0.05


def test_guide_pulls_a_flat_phrase_onto_the_melody():
    rate = 100.0
    classes = scale_pitch_classes(0, "major")

    def phrase(midis, cents):
        parts = []
        for midi in midis:
            parts.append(np.full(36, midi + cents / 100.0))
            parts.append(np.zeros(6))
        return midi_to_hz(np.concatenate(parts))

    melody = [60, 64, 67, 72]
    user = phrase(melody, -140)
    guide = phrase(melody, 0)
    from vocallift.align import map_guide

    mapped = map_guide(user, guide, rate, classes)
    followed = correct_pitch(user, rate, classes, strength=1, tightness=0.4, retune_ms=8, guide_midi=mapped)
    alone = correct_pitch(user, rate, classes, strength=1, tightness=0.4, retune_ms=8)
    assert _compress([int(round(n.target)) for n in followed.notes]) == melody
    assert _compress([int(round(n.target)) for n in alone.notes]) != melody
    assert residual_cents(followed) < 20


def test_monotone_follows_a_stretched_guide():
    from vocallift.align import map_guide

    rate = 100.0
    classes = scale_pitch_classes(0, "major")
    user = midi_to_hz(np.full(150, 58.5))
    guide = midi_to_hz(np.concatenate([np.full(50, m) for m in (60, 64, 67)]))
    mapped = map_guide(user, guide, rate, classes)
    result = correct_pitch(user, rate, classes, strength=1, tightness=0, retune_ms=5, guide_midi=mapped)
    assert [int(round(n.target)) for n in result.notes] == [60, 64, 67]


def test_harmony_is_a_third():
    f0 = np.array([0.0, 261.63, 261.63, 0.0])
    harmony = shifted_f0(f0, scale_pitch_classes(0, "major"), 2)
    assert abs(hz_to_midi(np.array([harmony[1]]))[0] - 64) < 0.05
    assert harmony[0] == 0


def test_master_limits_and_keeps_length():
    sr = 44100
    t = np.arange(sr) / sr
    y = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    out = master_vocal(y, sr, presence=0.7, reverb_amount=0.1)
    assert len(out) == len(y)
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 10 ** (-1 / 20) + 1e-3
    backing = render_backing(sr, sr, 0, "major")
    assert len(backing) == sr
    assert np.max(np.abs(backing)) < 0.35


def test_offkey_take_lands_closer_and_guide_keeps_the_melody():
    settings = EnhanceSettings(key="C", scale="major", strength=1, tightness=0.6, retune_ms=30, reverb=0.0)
    off = enhance(offkey_take(), settings=settings)
    assert off.before_cents > 25
    assert off.after_cents < off.before_cents * 0.55
    assert off.key_label == "C major"
    sung = _compress([int(round(n.target)) % 12 for n in off.correction.notes])
    assert sung == [m % 12 for m in MELODY]

    guided = enhance(flat_take(), settings=settings, guide=guide_take())
    targets = _compress([int(round(n.target)) % 12 for n in guided.correction.notes])
    assert targets == [m % 12 for m in MELODY]
    alone = enhance(flat_take(), settings=settings)
    assert _compress([int(round(n.target)) % 12 for n in alone.correction.notes]) != targets


def test_audio_roundtrip_and_loudness_shape():
    sr = 44100
    y = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    loaded = load_audio(to_wav_bytes(y, sr), filename="note.wav")
    assert abs(len(loaded) - sr) < 10
    assert frame_loudness_db(y, 512, 80).shape == (80,)


def _encode(y: np.ndarray, sr: int, fmt: str, codec: str) -> bytes:
    import av
    import io

    raw = io.BytesIO()
    container = av.open(raw, mode="w", format=fmt)
    stream = container.add_stream(codec, rate=sr)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(y.reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = sr
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return raw.getvalue()


def test_load_resamples_and_reads_browser_recordings():
    sr = 48000
    y = (0.2 * np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.3)) / sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    loaded = load_audio(buf.getvalue(), filename="note.wav")
    assert abs(len(loaded) - int(0.3 * 44100)) < 400
    for fmt, codec in (("webm", "libopus"), ("mp3", "mp3"), ("mp4", "aac")):
        decoded = load_audio(_encode(y, sr, fmt, codec))
        assert 0.15 * 44100 < len(decoded) < 0.6 * 44100
        assert float(np.max(np.abs(decoded))) > 0.01


def test_booth_demo_and_rejects_silence():
    from vocallift.api import app

    client = TestClient(app)
    page = client.get("/")
    assert page.status_code == 200
    assert "Land the note" in page.text
    assert client.get("/static/app.js").status_code == 200
    health = client.get("/api/health").json()
    assert health["ok"] is True

    demo = client.post("/api/demo", data={"mode": "offkey", "backing": "true"})
    assert demo.status_code == 200, demo.text
    body = demo.json()
    assert body["after_cents"] < body["before_cents"]
    assert body["mix_base64"]
    assert body["notes"]

    silent = io.BytesIO()
    sf.write(silent, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    silent.seek(0)
    rejected = client.post("/api/enhance", files={"file": ("silent.wav", silent.getvalue(), "audio/wav")})
    assert rejected.status_code == 400
