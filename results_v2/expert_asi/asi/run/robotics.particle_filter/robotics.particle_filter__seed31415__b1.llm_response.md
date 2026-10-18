```python
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Load the data
data_path = os.path.join('data', 'measurements.npz')
with np.load(data_path) as data:
    Y = data['Y'].astype(np.float64)
    t = data['t'].astype(np.float64)
    mask = data.get('mask', np.ones(len(t), dtype=bool))
    has_mask = mask is not None

# Determine time step
dt = t[1] - t[0]

# Initialize particles
num_particles = 1500
particles = np.random.randn(num_particles, 4) * 0.1  # Initial guess
log_weights = np.zeros(num_particles)

# Process each time step
results = {
    'state_estimates': [],
    'filter_diagnostics': [],
    'summary': defaultdict(float),
    'angle_wrap_count': 0,
}

for i in range(len(t)):
    # Update particles
    # (Implement particle update step here)
    pass

# Generate output files
os.makedirs('results', exist_ok=True)
np.savetxt('results/state_estimates.csv', results['state_estimates'], delimiter=',')
np.savetxt('results/filter_diagnostics.csv', results['filter_diagnostics'], delimiter=',')
with open('results/summary.json', 'w') as f:
    json.dump(results['summary'], f)
plt.savefig('results/overview.png')
```