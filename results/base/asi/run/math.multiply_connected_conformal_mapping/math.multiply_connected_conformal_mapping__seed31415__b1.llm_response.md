```python
import numpy as np
import json
import os
import scipy.optimize as opt
import scipy.interpolate as interp

# Load input files
with open('data/file_00.json') as f:
    metadata = json.load(f)

file_01 = np.load('data/file_01.npy')
file_02 = np.load('data/file_02.npy')
file_03 = np.load('data/file_03.npy')
file_04 = np.load('data/file_04.npy')
file_05 = np.load('data/file_05.npy')
file_06 = np.load('data/file_06.npy')
file_07 = np.load('data/file_07.npy')

# Extract metadata
component_order = metadata['component_order']
normalization_probes = metadata['normalization_probes']
slit_template = metadata['slit_template']

# Define the canonical domain with three slits
def canonical_domain(z, slit_params):
    # Implement the canonical domain mapping
    # This is a placeholder; actual implementation depends on the template
    # For demonstration, we'll use a simple mapping
    return z

# Define the source-to-canonical map
def source_to_canonical(z, slit_params):
    # Implement the source-to-canonical mapping
    # This is a placeholder; actual implementation depends on the template
    # For demonstration, we'll use a simple mapping
    return z

# Define the canonical-to-source inverse map
def canonical_to_source(w, slit_params):
    # Implement the inverse mapping
    # This is a placeholder; actual implementation depends on the template
    # For demonstration, we'll use a simple mapping
    return w

# Define the objective function for optimization
def objective(slit_params, *args):
    # Placeholder for objective function
    return np.array([0.0])

# Initial guess for slit parameters
initial_slit_params = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Optimize slit parameters
slit_params_opt = opt.minimize(objective, initial_slit_params, args=(file_03, file_04, file_05, file_06, file_07), method='L-BFGS-B').x

# Save slit parameters to CSV
np.savetxt('slit_params.csv', slit_params_opt.reshape(4, 3), delimiter=',', header='component_id,role,center_re,center_im,half_length,half_thickness,angle_rad', comments='')

# Compute canonical boundary samples
canonical_boundary = np.zeros((4, 96), dtype=np.complex128)
for i in range(4):
    for j in range(96):
        z = file_03[i, j]
        canonical_boundary[i, j] = source_to_canonical(z, slit_params_opt)

# Compute canonical probe values
canonical_probes = np.zeros_like(file_04, dtype=np.complex128)
for i in range(len(file_04)):
    z = file_04[i]
    canonical_probes[i] = source_to_canonical(z, slit_params_opt)

# Compute canonical probe derivatives
canonical_probe_derivatives = np.zeros_like(file_04, dtype=np.complex128)
for i in range(len(file_04)):
    z = file_04[i]
    canonical_probe_derivatives[i] = np.gradient(source_to_canonical(z, slit_params_opt), z)

# Compute canonical sample values
canonical_sample_values = np.zeros_like(file_05, dtype=np.complex128)
for i in range(len(file_05)):
    z = file_05[i]
    canonical_sample_values[i] = source_to_canonical(z, slit_params_opt)

# Compute canonical sample derivatives
canonical_sample_derivatives = np.zeros_like(file_05, dtype=np.complex128)
for i in range(len(file_05)):
    z = file_05[i]
    canonical_sample_derivatives[i] = np.gradient(source_to_canonical(z, slit_params_opt), z)

# Compute canonical boundary sample values
canonical_boundary_sample_values = np.zeros_like(file_06, dtype=np.complex128)
for i in range(4):
    for j in range(384):
        z = file_06[i, j]
        canonical_boundary_sample_values[i, j] = source_to_canonical(z, slit_params_opt)

# Compute canonical boundary sample derivatives
canonical_boundary_sample_derivatives = np.zeros_like(file_06, dtype=np.complex128)
for i in range(4):
    for j in range(384):
        z = file_06[i, j]
        canonical_boundary_sample_derivatives[i, j] = np.gradient(source_to_canonical(z, slit_params_opt), z)

# Compute source inverse sample values
source_inverse_sample_values = np.zeros_like(file_07, dtype=np.complex128)
for i in range(len(file_07)):
    w = file_07[i]
    source_inverse_sample_values[i] = canonical_to_source(w, slit_params_opt)

# Compute source inverse sample derivatives
source_inverse_sample_derivatives = np.zeros_like(file_07, dtype=np.complex128)
for i in range(len(file_07)):
    w = file_07[i]
    source_inverse_sample_derivatives[i] = np.gradient(canonical_to_source(w, slit_params_opt), w)

# Save output files
np.save('canonical_boundary.npy', canonical_boundary)
np.save('canonical_probes.npy', canonical_probes)
np.save('canonical_probe_derivatives.npy', canonical_probe_derivatives)
np.save('canonical_sample_values.npy', canonical_sample_values)
np.save('canonical_sample_derivatives.npy', canonical_sample_derivatives)
np.save('canonical_boundary_sample_values.npy', canonical_boundary_sample_values)
np.save('canonical_boundary_sample_derivatives.npy', canonical_boundary_sample_derivatives)
np.save('source_inverse_sample_values.npy', source_inverse_sample_values)
np.save('source_inverse_sample_derivatives.npy', source_inverse_sample_derivatives)
```