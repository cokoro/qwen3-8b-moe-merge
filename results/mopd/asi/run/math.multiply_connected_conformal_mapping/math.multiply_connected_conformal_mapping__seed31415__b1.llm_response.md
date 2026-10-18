```python
import numpy as np
import scipy.optimize as optimize
import json

# Read metadata
with open('data/file_00.json') as f:
    metadata = json.load(f)

# Read input files
int32_indices = np.load('data/file_01.npy')
complex_coefficients = np.load('data/file_02.npy')
source_boundary_samples = np.load('data/file_03.npy')
source_probes = np.load('data/file_04.npy')
source_samples = np.load('data/file_05.npy')
source_boundary_samples_dense = np.load('data/file_06.npy')
canonical_samples = np.load('data/file_07.npy')

# Define the slit-like template formula
def slit_template(center, half_length, half_thickness, angle):
    # Implement the slit-like template formula here
    pass

# Define the source-to-canonical map
def source_to_canonical(z):
    # Implement the source-to-canonical map here
    pass

# Define the canonical-to-source inverse map
def canonical_to_source(w):
    # Implement the canonical-to-source inverse map here
    pass

# Optimize the map parameters
# This is a placeholder for the actual optimization algorithm
parameters = optimize.minimize(lambda params: 0, [0.0]*len(metadata['slit_params']))

# Generate output files
np.savetxt('slit_params.csv', metadata['slit_params'], delimiter=',', fmt='%s')
np.save('canonical_boundary.npy', source_to_canonical(source_boundary_samples))
np.save('canonical_probes.npy', source_to_canonical(source_probes))
np.save('canonical_probe_derivatives.npy', source_to_canonical(source_probes))
np.save('canonical_sample_values.npy', source_to_canonical(source_samples))
np.save('canonical_sample_derivatives.npy', source_to_canonical(source_samples))
np.save('canonical_boundary_sample_values.npy', source_to_canonical(source_boundary_samples_dense))
np.save('canonical_boundary_sample_derivatives.npy', source_to_canonical(source_boundary_samples_dense))
np.save('source_inverse_sample_values.npy', canonical_to_source(canonical_samples))
np.save('source_inverse_sample_derivatives.npy', canonical_to_source(canonical_samples))
```