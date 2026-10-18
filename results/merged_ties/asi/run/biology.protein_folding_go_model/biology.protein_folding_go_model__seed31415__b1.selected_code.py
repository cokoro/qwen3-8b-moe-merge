```bash
#!<abs_path>

set -e

cd $(dirname $0)

source activate openmm

python simulation.py
```