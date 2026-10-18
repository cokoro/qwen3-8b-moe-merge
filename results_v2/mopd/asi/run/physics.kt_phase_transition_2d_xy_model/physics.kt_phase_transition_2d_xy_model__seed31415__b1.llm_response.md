```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import random
import math

# Load configuration
with open('data/control_primary.json') as f:
    control = json.load(f)

# Load data
scan_axis_a = np.load('data/scan_axis_a.npy')
scan_axis_b = np.load('data/scan_axis_b.npy')
probe_axis_a = np.load('data/probe_axis_a.npy')
state_tokens = np.load('data/state_tokens.npz')

# Initialize simulation
L = scan_axis_b[-1]
T = scan_axis_a[-1]
theta = state_tokens[control['state_bundle']['file']][0]
# Perform hybrid Monte Carlo simulation
# ...

# Compute helicity modulus
# ...

# Compute vortex density
# ...

# Compute radial correlation profiles
# ...

# Compute second-moment correlation-length ratio
# ...

# Compute magnetic susceptibility
# ...

# Perform finite-size analysis
# ...

# Compute phase summary
# ...

# Compute phase fit diagnostics
# ...

# Plot overview
# ...
```