```bash
from scipy.interpolate import interp1d
import numpy as np

# Load data
with open('data/file_00.json') as f:
    metadata = json.load(f)
    component_count = metadata['component_count']
    normalization = metadata['normalization']
    slit_template = metadata['slit_template']

file_01 = np.load('data/file_01.npy')
file_02 = np.load('data/file_02.npy')
file_03 = np.load('data/file_03.npy')
file_04 = np.load('data/file_04.npy')
file_05 = np.load('data/file_05.npy')
file_06 = np.load('data/file_06.npy')
file_07 = np.load('data/file_07.npy')

# Build analytic map
def analytic_map(z, params):
    # Implement analytic map using Fourier coefficients and slit template
    return np.sum(file_02[file_01] * np.exp(1j * params[file_01] * np.arctan2(z.imag, z.real)), axis=0)

# Solve for parameters
params = np.zeros(component_count)
for i in range(component_count):
    params[i] = np.argmin(np.abs(analytic_map(file_04, params) - normalization[file_04]))

# Evaluate map
canonical_boundary = analytic_map(file_06, params)
canonical_probes = analytic_map(file_04, params)
canonical_samples = analytic_map(file_05, params)
canonical_inverse_samples = np.conj(canonical_samples) / np.conj(analytic_map(canonical_samples, params))

# Save outputs
np.savetxt('slit_params.csv', np.array([[0, 0.0, 0.0, 1.0, 1.0, 0.0],
                                        [1, 0.0, 0.0, 1.0, 1.0, 0.0],
                                        [2, 0.0, 0.0, 1.0, 1.0, 0.0],
                                        [3, 0.0, 0.0, 1.0, 1.0, 0.0]]),
           delimiter=',', header='component_id,center_re,center_im,half_length,half_thickness,angle_rad', comments='')
np.save('canonical_boundary.npy', canonical_boundary)
np.save('canonical_probes.npy', canonical_probes)
np.save('canonical_probe_derivatives.npy', np.gradient(canonical_probes, axis=0))
np.save('canonical_sample_values.npy', canonical_samples)
np.save('canonical_sample_derivatives.npy', np.gradient(canonical_samples, axis=0))
np.save('canonical_boundary_sample_values.npy', analytic_map(file_06, params))
np.save('canonical_boundary_sample_derivatives.npy', np.gradient(analytic_map(file_06, params), axis=1))
np.save('source_inverse_sample_values.npy', canonical_inverse_samples)
np.save('source_inverse_sample_derivatives.npy', np.gradient(canonical_inverse_samples, axis=0))
```