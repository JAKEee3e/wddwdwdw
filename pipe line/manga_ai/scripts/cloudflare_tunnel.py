from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TunnelProcess:
    process: subprocess.Popen
    public_url: str


def _default_cloudflared_path(cache_dir: Path) -> Path:
    exe = "cloudflared.exe" if os.name == "nt" else "cloudflared"
    return cache_dir / exe


def ensure_cloudflared(cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = _default_cloudflared_path(cache_dir)
    if path.exists():
        return path

    if os.name == "nt":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif sys.platform == "linux":
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    else:
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"

    urllib.request.urlretrieve(url, path)

    try:
        os.chmod(path, 0o755)
    except Exception:
        pass

    return path


def start_tunnel(
    local_port: int,
    cache_dir: str | Path,
    host: str = "127.0.0.1",
    max_wait_s: int = 45,
) -> TunnelProcess:
    cloudflared = ensure_cloudflared(cache_dir)

    cmd = [str(cloudflared), "tunnel", "--url", f"http://{host}:{local_port}", "--no-autoupdate"]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    url_re = re.compile(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com")
    public_url: Optional[str] = None

    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        if proc.poll() is not None:
            break
        if proc.stdout is None:
            break
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        m = url_re.search(line)
        if m:
            public_url = m.group(0)
            break

    if not public_url:
        try:
            proc.terminate()
        except Exception:
            pass
        raise RuntimeError("Failed to start Cloudflare tunnel (no public URL found in output)")

    return TunnelProcess(process=proc, public_url=public_url)
