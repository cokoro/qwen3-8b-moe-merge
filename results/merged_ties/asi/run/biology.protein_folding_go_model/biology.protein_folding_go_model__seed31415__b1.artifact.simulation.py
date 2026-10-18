```bash
#!/bin/bash

set -e

cd $(dirname $0)

source activate openmm

python simulation.py
```