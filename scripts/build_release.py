"""Freeze the booth into a folder and pack it for GitHub Releases."""

from __future__ import annotations

import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vocallift import __version__  # noqa: E402

NOTES = """VocalLift {version}

Unzip this folder and keep it together.

Windows: double-click VocalLift.exe
macOS and Linux: open a terminal here and run ./VocalLift

The booth opens at http://127.0.0.1:8000

Command line, from this folder:
  VocalLift enhance take.wav out.wav --key C --scale major
  VocalLift skin build reading.wav --owner "Your Name" --consent
  VocalLift skin speak line.wav out.wav --voice "Your Name"

This build is the booth, the signal engine, and voice skins with both
engines. PyTorch is included. The neural engine downloads its weights
(1.3 GB) once, from the booth's download button or with
  VocalLift skin neural-download
Training the resynthesizer stays in the Colab notebook. When you have a checkpoint, put it
at checkpoints/vocallift.pt next to the program and turn on Neural resynth.

The download is unsigned. Windows SmartScreen and macOS Gatekeeper ask you
to confirm the first launch. On a Mac, right-click VocalLift and choose Open.
"""


def _platform_tag() -> tuple[str, str]:
    system = {"linux": "linux", "darwin": "macos", "windows": "windows"}[platform.system().lower()]
    machine = platform.machine().lower()
    arch = {"x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)
    return system, arch


def _freeze() -> Path:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("The release bundles the neural voice-skin engine. Install CPU PyTorch first (see the release workflow).") from exc
    print(f"bundling torch {torch.__version__}")
    dist = ROOT / "dist"
    work = ROOT / "build"
    shutil.rmtree(dist, ignore_errors=True)
    shutil.rmtree(work, ignore_errors=True)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--name",
        "VocalLift",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        "--paths",
        str(ROOT),
        "--add-data",
        f"{ROOT / 'web'}{os.pathsep}web",
        "--collect-all",
        "av",
        "--collect-all",
        "soundfile",
        "--collect-all",
        "pyworld",
        "--collect-submodules",
        "uvicorn",
        "--collect-submodules",
        "fastapi",
        "--hidden-import",
        "multipart",
        "--hidden-import",
        "vocallift.skin",
        "--hidden-import",
        "vocallift.skin_neural",
        "--collect-submodules",
        "vocallift.vendor",
        "--hidden-import",
        "pkg_resources",
        "--copy-metadata",
        "pyworld",
        "--exclude-module",
        "torchaudio",
        "--exclude-module",
        "torchvision",
        "--exclude-module",
        "tensorboard",
        "--exclude-module",
        "librosa",
        "--exclude-module",
        "numba",
        "--exclude-module",
        "IPython",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "nbformat",
        "--exclude-module",
        "matplotlib",
        "--exclude-module",
        "tkinter",
        str(ROOT / "packaging" / "entry.py"),
    ]
    subprocess.check_call(cmd, cwd=ROOT)
    folder = dist / "VocalLift"
    if not folder.is_dir():
        raise SystemExit(f"PyInstaller did not write {folder}")
    return folder


def _mark_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _pack(folder: Path) -> Path:
    system, arch = _platform_tag()
    (folder / "README.txt").write_text(NOTES.format(version=__version__), encoding="utf-8")
    binary = folder / ("VocalLift.exe" if system == "windows" else "VocalLift")
    if not binary.is_file():
        raise SystemExit(f"Missing binary {binary}")
    _mark_executable(binary)
    if system == "macos":
        opener = folder / "Open VocalLift.command"
        opener.write_text("#!/bin/bash\ncd \"$(dirname \"$0\")\"\n./VocalLift\n", encoding="utf-8")
        _mark_executable(opener)
    out = ROOT / "dist" / "release"
    out.mkdir(parents=True)
    base = f"VocalLift-{__version__}-{system}-{arch}"
    if system == "windows":
        archive = out / f"{base}.zip"
        if archive.exists():
            archive.unlink()
        shutil.make_archive(str(out / base), "zip", root_dir=folder.parent, base_dir=folder.name)
    else:
        archive = out / f"{base}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=folder.name)
    print(archive)
    return archive


def main() -> None:
    folder = _freeze()
    _pack(folder)


if __name__ == "__main__":
    main()
