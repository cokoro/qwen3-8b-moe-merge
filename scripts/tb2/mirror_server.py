"""Local stand-in for the GitHub release downloads that TB2 verifiers need (github.com is unreachable here).

Serves /root/gpufree-data/tb2/mirror on 127.0.0.1:18080 and fills misses:
  /astral-sh/uv/releases/download/<ver>/uv-x86_64-unknown-linux-<gnu|musl>.tar.gz
      built on demand from the official uv wheel on the tuna PyPI mirror (same binaries), then cached
  /python-build-standalone/<release>/<file>
      302 redirect to the npmmirror copy of the GitHub release
"""

import io
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import zipfile
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote

ROOT = Path("/root/gpufree-data/tb2/mirror")
PIP = "/opt/conda/bin/pip"
PYPI = "https://pypi.tuna.tsinghua.edu.cn/simple"
NPM_PBS = "https://registry.npmmirror.com/-/binary/python-build-standalone"
UV_RE = re.compile(r"^/astral-sh/uv/releases/download/([0-9][0-9A-Za-z.\-]*)/uv-x86_64-unknown-linux-(gnu|musl)\.tar\.gz$")
PBS_RE = re.compile(r"^/python-build-standalone/([0-9]+)/([^/]+)$")
_build_lock = threading.Lock()


def build_uv_tarball(version: str, libc: str, out: Path) -> None:
    platform = "manylinux_2_17_x86_64" if libc == "gnu" else "musllinux_1_1_x86_64"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [PIP, "download", f"uv=={version}", "--no-deps", "--only-binary=:all:",
             "--platform", platform, "-d", tmp, "-i", PYPI, "-q"],
            check=True, timeout=600,
        )
        whl = next(Path(tmp).glob("*.whl"))
        name = f"uv-x86_64-unknown-linux-{libc}"
        buf = io.BytesIO()
        with zipfile.ZipFile(whl) as zf, tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for binary in ("uv", "uvx"):
                data = zf.read(f"uv-{version}.data/scripts/{binary}")
                info = tarfile.TarInfo(f"{name}/{binary}")
                info.size, info.mode = len(data), 0o755
                tf.addfile(info, io.BytesIO(data))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_out = out.with_suffix(".part")
        tmp_out.write_bytes(buf.getvalue())
        shutil.move(tmp_out, out)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(self.path.split("?", 1)[0])
        local = ROOT / path.lstrip("/")
        if not local.is_file():
            if m := UV_RE.match(path):
                with _build_lock:
                    if not local.is_file():
                        try:
                            build_uv_tarball(m.group(1), m.group(2), local)
                            self.log_message("built %s", local)
                        except Exception as e:  # unknown version etc.
                            self.log_message("uv build failed for %s: %s", path, e)
            elif m := PBS_RE.match(path):
                self.send_response(302)
                self.send_header("Location", f"{NPM_PBS}/{m.group(1)}/{quote(m.group(2))}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        super().do_GET()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18080), partial(Handler, directory=str(ROOT))).serve_forever()
