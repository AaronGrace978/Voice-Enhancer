"""Command line: enhance a file, wear a voice skin, train the resynthesizer, or open the booth."""

from __future__ import annotations

import argparse
import os
import sys

from vocallift.audio import load_audio, save_audio
from vocallift.chords import mix_with_backing
from vocallift.pipeline import EnhanceSettings, enhance


def _build() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vocallift", description="VocalLift singing engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    enhance_cmd = sub.add_parser("enhance", help="Write an enhanced vocal")
    enhance_cmd.add_argument("input")
    enhance_cmd.add_argument("output")
    enhance_cmd.add_argument("--guide", help="A take of the melody to follow")
    enhance_cmd.add_argument("--key", default="auto")
    enhance_cmd.add_argument("--scale", default="auto")
    enhance_cmd.add_argument("--strength", type=float, default=0.85)
    enhance_cmd.add_argument("--tightness", type=float, default=0.4)
    enhance_cmd.add_argument("--retune-ms", type=float, default=50)
    enhance_cmd.add_argument("--vibrato", type=float, default=1.0)
    enhance_cmd.add_argument("--add-vibrato", type=float, default=0.0)
    enhance_cmd.add_argument("--reverb", type=float, default=0.14)
    enhance_cmd.add_argument("--presence", type=float, default=0.65)
    enhance_cmd.add_argument("--harmony", action="store_true")
    enhance_cmd.add_argument("--no-repair", action="store_true")
    enhance_cmd.add_argument("--backing", action="store_true", help="Also write a chord-bed mix")
    enhance_cmd.add_argument("--neural", action="store_true")
    enhance_cmd.add_argument("--fast", action="store_true")

    serve = sub.add_parser("serve", help="Open the vocal booth")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    train = sub.add_parser("train", help="Train or fine-tune the neural resynthesizer")
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--finetune-steps", type=int, default=80)
    train.add_argument("--out", default="checkpoints/vocallift.pt")
    train.add_argument("--finetune", nargs="*", default=[], help="Recordings of the voice to adapt to")
    train.add_argument("--init", default="", help="Checkpoint to keep training")

    skin = sub.add_parser("skin", help="Voice skins: let other people speak in a voice its owner shared")
    skin_sub = skin.add_subparsers(dest="skin_cmd", required=True)
    build = skin_sub.add_parser("build", help="Learn a voice from its owner's recordings")
    build.add_argument("recordings", nargs="+", help="Recordings of the owner speaking (a few minutes in total)")
    build.add_argument("--owner", required=True, help="Whose voice this is")
    build.add_argument("--consent", action="store_true", help="Confirm this is your voice, or its owner agreed to share it")
    build.add_argument("--terms", default="", help="The owner's terms for using the voice, stored with the skin")
    build.add_argument("--out", default="", help="Skin folder (default: voices/<owner>)")
    build.add_argument("--fast", action="store_true")
    speak = skin_sub.add_parser("speak", help="Say what a recording says, in a skin's voice")
    speak.add_argument("input")
    speak.add_argument("output")
    speak.add_argument("--voice", required=True, help="Skin name or folder")
    speak.add_argument("--engine", default="signal", choices=["signal", "neural"])
    speak.add_argument("--amount", type=float, default=1.0, help="0 keeps the speaker, 1 is all owner")
    speak.add_argument("--pitch", default="match", choices=["match", "keep"], help="keep leaves a sung melody alone")
    speak.add_argument("--shift", type=float, default=0.0, help="Extra semitones")
    speak.add_argument("--detail", type=float, default=0.5, help="Signal engine: owner frames vs. warped speaker")
    speak.add_argument("--fast", action="store_true")
    skin_sub.add_parser("list", help="Show the skins in voices/")
    return parser


def _skin(args) -> None:
    from vocallift.skin import (
        SkinSettings,
        build_skin,
        find_skin,
        list_skins,
        load_recordings,
        save_skin,
        save_tagged,
        slug,
        speak_as,
        voices_dir,
    )

    if args.skin_cmd == "list":
        skins = list_skins()
        if not skins:
            print(f"No skins in {voices_dir()} yet.")
        for item in skins:
            print(f"{item.name}: {item.meta.owner}, {item.meta.seconds:.0f}s of audio, around {item.meta.median_hz:.0f} Hz")
        return
    if args.skin_cmd == "build":
        recordings = load_recordings(args.recordings)
        try:
            built = build_skin(recordings, owner=args.owner, consent=args.consent, terms=args.terms, fast=args.fast)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        path = save_skin(built, args.out or voices_dir() / slug(args.owner), recordings)
        print(f"saved {path}  {built.meta.seconds:.0f}s of {built.meta.owner}, around {built.meta.median_hz:.0f} Hz")
        return
    try:
        voice = find_skin(args.voice)
        settings = SkinSettings(amount=args.amount, pitch=args.pitch, shift=args.shift, detail=args.detail, fast=args.fast)
        result = speak_as(load_audio(args.input), voice, settings=settings, engine=args.engine)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    save_tagged(args.output, result.audio, result.sr, result.owner)
    print(result.summary())


def _open_booth(url: str, host: str) -> None:
    if os.environ.get("CI") or os.environ.get("VOCALLIFT_NO_BROWSER"):
        return
    if not getattr(sys, "frozen", False):
        return
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return
    import threading
    import webbrowser

    def _open() -> None:
        try:
            webbrowser.open(url)
        except Exception:
            return

    threading.Timer(0.6, _open).start()


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and getattr(sys, "frozen", False):
        argv = ["serve"]
    args = _build().parse_args(argv)
    if args.cmd == "serve":
        import uvicorn

        from vocallift.api import app

        url = f"http://{args.host}:{args.port}"
        print(f"VocalLift booth at {url}")
        _open_booth(url, args.host)
        uvicorn.run(app, host=args.host, port=args.port)
        return
    if args.cmd == "skin":
        _skin(args)
        return
    if args.cmd == "train":
        try:
            from vocallift.neural.infer import load_model
            from vocallift.neural.train import finetune_files, save_checkpoint, train_steps
        except ImportError as exc:
            raise SystemExit("Training needs PyTorch. Open the Colab notebook, or pip install torch torchaudio.") from exc

        if args.init:
            model = load_model(args.init)
            losses = []
        else:
            model, losses = train_steps(steps=args.steps)
        if args.finetune:
            losses = finetune_files(model, args.finetune, steps=args.finetune_steps)
        save_checkpoint(model, args.out)
        if losses:
            print(f"saved {args.out}  final loss {losses[-1]:.4f}")
        else:
            print(f"saved {args.out}")
        return

    settings = EnhanceSettings(
        key=args.key,
        scale=args.scale,
        strength=args.strength,
        tightness=args.tightness,
        retune_ms=args.retune_ms,
        vibrato=args.vibrato,
        add_vibrato=args.add_vibrato,
        reverb=args.reverb,
        presence=args.presence,
        harmony=args.harmony,
        repair=not args.no_repair,
        neural=args.neural,
        fast=args.fast,
    )
    audio = load_audio(args.input)
    guide = load_audio(args.guide) if args.guide else None
    result = enhance(audio, settings=settings, guide=guide)
    save_audio(args.output, result.audio, result.sr)
    print(result.summary())
    if args.backing:
        mix_path = str(args.output).rsplit(".", 1)[0] + ".mix.wav"
        mix = mix_with_backing(result.audio, result.sr, result.key_root, result.scale, result.detected_mode)
        save_audio(mix_path, mix, result.sr)
        print(f"mix {mix_path}")


if __name__ == "__main__":
    main()
