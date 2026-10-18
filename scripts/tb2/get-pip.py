# Minimal get-pip replacement served on the local mirror (bootstrap.pypa.io is unreachable; the aliyun copy is stale).
# Installs the official pip 25.0.1 wheel (py3.8+) into the interpreter running this script (harbor's server venv).
import os, sys, sysconfig, tempfile, urllib.request
WHL = "pip-25.0.1-py3-none-any.whl"
dst = os.path.join(tempfile.mkdtemp(), WHL)
urllib.request.urlretrieve("http://127.0.0.1:18080/" + WHL, dst)
try:
    import distutils.cmd  # noqa: F401
except Exception:
    # Debian/Ubuntu python<3.10 without python3-distutils: pip only works with its sysconfig backend there.
    # sysconfig._PIP_USE_SYSCONFIG is pip's distributor switch; set it for this venv only via a .pth file.
    sysconfig._PIP_USE_SYSCONFIG = True
    purelib = sysconfig.get_paths()["purelib"]
    os.makedirs(purelib, exist_ok=True)
    with open(os.path.join(purelib, "_harbor_pip_sysconfig.pth"), "w") as f:
        f.write("import sysconfig; sysconfig._PIP_USE_SYSCONFIG = True\n")
sys.path.insert(0, dst)
from pip._internal.cli.main import main  # noqa: E402
sys.exit(main(["install", "--no-index", "--quiet", dst] + sys.argv[1:]))
