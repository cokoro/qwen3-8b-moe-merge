#!/bin/bash
# Local patches to harbor 0.24.0 (uv tool install) needed on this network-restricted host. Idempotent.
H=/root/.local/share/uv/tools/harbor/lib/python3.12/site-packages/harbor/environments/singularity
# 1) container bootstrap installs python/tmux/fastapi through mirrors; can exceed the hard-coded 60s server wait
sed -i "s/            for i in range(60):$/            for i in range(300):  # local patch: bootstrap via mirrors can exceed 60s/" $H/singularity.py
# 2) bootstrap.pypa.io is unreachable; serve get-pip.py (from mirrors.aliyun.com/pypi/get-pip.py) on the local mirror
sed -i "s#https://bootstrap.pypa.io/get-pip.py#http://127.0.0.1:18080/get-pip.py#" $H/bootstrap.sh
grep -n "range(300)" $H/singularity.py; grep -n "get-pip.py'" $H/bootstrap.sh
# 3) server.py forces PATH="/usr/bin:/usr/local/bin:$PATH", which makes python:3.x images resolve python3 to an
#    apt-installed /usr/bin/python3 instead of the image's /usr/local/bin/python3 (docker env uses the image PATH).
# 4) bootstrap.sh hard-codes /usr/bin/python3 and apt-installs a second interpreter when the image ships its own python3.
python3 - "$H" <<'PYEOF'
import sys
H = sys.argv[1]
def patch(path, old, new):
    s = open(path).read()
    if new in s:
        return
    assert old in s, (path, old)
    open(path, "w").write(s.replace(old, new))
patch(f"{H}/server.py",
      'env["PATH"] = "/usr/bin:/usr/local/bin:" + env.get("PATH", "/bin")',
      'env["PATH"] = env.get("PATH") or "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"  # local patch: keep image PATH (docker semantics)')
patch(f"{H}/bootstrap.sh",
      "_SYS_PY=/usr/bin/python3\n",
      "_SYS_PY=$(command -v python3 2>/dev/null || echo /usr/bin/python3)  # local patch: reuse image python3\n")
PYEOF
grep -n "local patch" $H/server.py $H/bootstrap.sh
# (python<3.10 images without distutils are handled in mirror/get-pip.py, inside the server venv only.)
sed -i "s/_PIP_USE_SYSCONFIG=1 //g" $H/bootstrap.sh
