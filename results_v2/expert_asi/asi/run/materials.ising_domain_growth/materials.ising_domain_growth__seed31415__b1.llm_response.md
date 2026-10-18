```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def simulate_ising(params):
    # Load parameters
    grid_size = params['grid_size']
    control = params['control']
    n_steps = params['n_steps']
    n_snapshots = params['n_snapshots']
    n_measurements = params['n_measurements']
    snapshot_times = params['snapshot_times

```