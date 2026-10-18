```bash
#!/bin/bash

set -e

SCRIPT_DIR="$(dirname "$0")"

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install -e .
python3 "$SCRIPT_DIR/analysis.py"
deactivate
rm -rf venv
```