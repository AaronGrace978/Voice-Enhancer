"""Start a frozen booth and check that the demo actually runs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def main() -> None:
    binary = Path(sys.argv[1]).resolve()
    port = "8765"
    log_path = binary.parent / "booth-smoke.log"
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env["CI"] = "1"
    env["VOCALLIFT_NO_BROWSER"] = "1"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(binary), "serve", "--host", "127.0.0.1", "--port", port],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        health = None
        for _ in range(60):
            if proc.poll() is not None:
                break
            try:
                with urllib.request.urlopen(base + "/api/health", timeout=2) as resp:
                    health = json.loads(resp.read().decode())
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.5)
        if not health or not health.get("ok"):
            raise SystemExit(f"health check failed\n{log_path.read_text(encoding='utf-8', errors='replace')}")
        print("health", health)
        with urllib.request.urlopen(base + "/", timeout=10) as resp:
            page = resp.read().decode()
        if "Open in Colab" not in page:
            raise SystemExit("booth page did not include the Colab link")
        body = b"mode=offkey&backing=false"
        request = urllib.request.Request(
            base + "/api/demo",
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(request, timeout=180) as resp:
            payload = json.loads(resp.read().decode())
        print("cents", payload.get("before_cents"), payload.get("after_cents"))
        if not (payload["before_cents"] > payload["after_cents"]):
            raise SystemExit(f"demo did not improve the take: {payload.get('before_cents')} -> {payload.get('after_cents')}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
