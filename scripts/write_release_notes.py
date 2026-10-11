"""Write GitHub release notes for the tag in GITHUB_REF_NAME."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    ref = os.environ.get("GITHUB_REF_NAME", "v0.1.0")
    version = ref[1:] if ref.startswith("v") else ref
    text = f"""VocalLift {ref} for Windows, macOS, and Linux.

Unzip the archive and keep the folder together.

- **Windows:** double-click `VocalLift.exe`
- **macOS Apple Silicon:** `VocalLift-{version}-macos-arm64.tar.gz`, then `./VocalLift` or double-click `Open VocalLift.command`
- **macOS Intel:** `VocalLift-{version}-macos-x64.tar.gz`
- **Linux:** `VocalLift-{version}-linux-x64.tar.gz`, then `./VocalLift`

The booth opens at http://127.0.0.1:8000. These builds are unsigned, so Windows SmartScreen and macOS Gatekeeper ask you to confirm the first launch.

The download is the booth and the signal engine. Train the resynthesizer in the [Colab notebook](https://colab.research.google.com/github/AaronGrace978/Voice-Enhancer/blob/{ref}/notebooks/VocalLift_Colab.ipynb), then put `checkpoints/vocallift.pt` next to the program.
"""
    Path("release-notes.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
