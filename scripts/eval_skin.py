"""Measure a voice skin on CMU ARCTIC: does a converted voice land nearer the owner?

Downloads two ARCTIC speakers (slt, a woman; bdl, a man), builds a skin from
100 of slt's sentences, converts 15 of bdl's sentences that are not in the
skin, and compares each against slt saying the same sentence.

Mel-cepstral distortion (MCD, in dB, after DTW alignment) is the standard
voice-conversion distance. Lower is closer.

    python scripts/eval_skin.py [--engine signal|neural] [--asr] [--keep out_dir]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyworld as pw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vocallift.audio import load_audio, save_audio, resample  # noqa: E402
from vocallift.skin import SkinSettings, build_skin, convert  # noqa: E402

URL = "http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_{spk}_arctic/wav/arctic_a{n:04d}.wav"
SR = 16000


def fetch(cache: Path, spk: str, n: int) -> Path:
    path = cache / spk / f"arctic_a{n:04d}.wav"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(spk=spk, n=n), path)
    return path


def mcep(y: np.ndarray, sr: int) -> np.ndarray:
    x = y.astype(np.float64)
    f0, t = pw.dio(x, sr, frame_period=5.0)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)
    code = pw.code_spectral_envelope(sp, sr, 25)
    energy = 10 * np.log10(np.sum(sp, axis=1) + 1e-12)
    return code[energy > energy.max() - 40][:, 1:]


def dtw_mcd(a: np.ndarray, b: np.ndarray) -> float:
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    n, m = cost.shape
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] = row[j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    i, j, steps = n, m, 0
    while i > 0 and j > 0:
        steps += 1
        move = np.argmin([acc[i - 1, j - 1], acc[i - 1, j], acc[i, j - 1]])
        i, j = (i - 1, j - 1) if move == 0 else (i - 1, j) if move == 1 else (i, j - 1)
    # Same scale as the usual 10 / ln10 * sqrt(2 * sum) on mel-cepstra.
    return float(10.0 / np.log(10) * np.sqrt(2.0) * acc[n, m] / steps)


class Recognizer:
    """wav2vec 2.0 (LibriSpeech 960 h) greedy transcripts, to check the words survive."""

    def __init__(self) -> None:
        import torchaudio

        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().eval()
        self.labels = bundle.get_labels()
        text = urllib.request.urlopen("http://festvox.org/cmu_arctic/cmuarctic.data").read().decode()
        self.reference = dict(re.findall(r'\( arctic_(a\d+) "(.*?)" \)', text))

    def transcribe(self, y: np.ndarray, sr: int) -> list[str]:
        import torch

        x = torch.from_numpy(resample(y, sr, 16000))[None]
        with torch.inference_mode():
            ids = self.model(x)[0][0].argmax(-1).tolist()
        chars, previous = [], None
        for index in ids:
            if index != previous and self.labels[index] != "-":
                chars.append(self.labels[index])
            previous = index
        return "".join(chars).replace("|", " ").lower().split()

    def wer(self, y: np.ndarray, sr: int, n: int) -> float:
        ref = re.sub(r"[^a-z' ]", "", self.reference[f"a{n:04d}"].lower()).split()
        hyp = self.transcribe(y, sr)
        row = list(range(len(hyp) + 1))
        for i in range(1, len(ref) + 1):
            new = [i] + [0] * len(hyp)
            for j in range(1, len(hyp) + 1):
                new[j] = min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (ref[i - 1] != hyp[j - 1]))
            row = new
        return row[-1] / max(1, len(ref))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="signal", choices=["signal", "neural"])
    parser.add_argument("--cache", default=".cache/arctic")
    parser.add_argument("--keep", default="")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--asr", action="store_true", help="Also measure word error rate with wav2vec 2.0")
    args = parser.parse_args()
    cache = Path(args.cache)
    asr = Recognizer() if args.asr else None
    wer_before, wer_after = [], []

    owner = [load_audio(fetch(cache, "slt", n), sr=SR) for n in range(1, 101)]
    tests = list(range(101, 116))
    started = time.perf_counter()
    if args.engine == "neural":
        from vocallift.skin_neural import NeuralSkin

        neural = NeuralSkin.from_recordings(owner, SR)
    else:
        skin = build_skin(owner, sr=SR, owner="slt (CMU ARCTIC)", consent=True, fast=args.fast)
    print(f"skin built in {time.perf_counter() - started:.1f}s")

    before, after, floor = [], [], []
    for n in tests:
        src = load_audio(fetch(cache, "bdl", n), sr=SR)
        ref = load_audio(fetch(cache, "slt", n), sr=SR)
        if args.engine == "neural":
            out = neural.convert(src, SR, out_sr=SR)
        else:
            out = convert(src, skin, sr=SR, settings=SkinSettings(fast=args.fast), out_sr=SR).audio
        if args.keep:
            Path(args.keep).mkdir(parents=True, exist_ok=True)
            save_audio(Path(args.keep) / f"a{n:04d}_bdl.wav", src, SR)
            save_audio(Path(args.keep) / f"a{n:04d}_as_slt.wav", out, SR)
            save_audio(Path(args.keep) / f"a{n:04d}_slt.wav", ref, SR)
        target = mcep(ref, SR)
        before.append(dtw_mcd(mcep(src, SR), target))
        after.append(dtw_mcd(mcep(out, SR), target))
        # What the same pipeline costs with nothing to convert: slt resynthesized through her own skin.
        if args.engine == "signal" and n == tests[0]:
            floor.append(dtw_mcd(mcep(convert(ref, skin, sr=SR, out_sr=SR).audio, SR), target))
        line = f"a{n:04d}  bdl→slt MCD {before[-1]:5.2f} dB  converted {after[-1]:5.2f} dB"
        if asr:
            wer_before.append(asr.wer(src, SR, n))
            wer_after.append(asr.wer(out, SR, n))
            line += f"  WER {100 * wer_before[-1]:4.0f}% → {100 * wer_after[-1]:4.0f}%"
        print(line)
    print(f"\nmean MCD to slt: original bdl {np.mean(before):.2f} dB, converted {np.mean(after):.2f} dB ({args.engine})")
    if asr:
        print(f"word error rate: original bdl {100 * np.mean(wer_before):.1f}%, converted {100 * np.mean(wer_after):.1f}%")
    if floor:
        print(f"slt through her own skin: {np.mean(floor):.2f} dB")
    print(f"total {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
