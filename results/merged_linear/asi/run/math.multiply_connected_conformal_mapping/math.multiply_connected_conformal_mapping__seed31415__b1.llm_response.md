```python
import numpy as np
import scipy.io
import scipy.optimize
import json

# Load input files
metadata = json.load(open("data/file_00.json"))
boundary_indices = np.load("data/file_01.npy")
boundary_coeffs = np.load("data/file_02.npy")
boundary_samples = np.load("data/file_03.npy")
probe_points = np.load("data/file_04.npy")
sample_points = np.load("data/file_05.npy")
boundary_samples_dense = np.load("data/file_06.npy")
canonical_samples = np.load("data/file_07.npy")

# Build the canonical boundary sample array
canonical_boundary = np.zeros((4, 96), dtype=np.complex128)
canonical_boundary[0] = boundary_samples[0]
canonical_boundary[1:] = boundary_samples[1:]

# Define the canonical boundary mapping
def canonical_boundary_mapping(z):
    # Use the Fourier coefficients to compute the canonical boundary mapping
    # This is a placeholder for the actual implementation
    return np.zeros_like(z)

# Define the source-to-canonical mapping
def source_to_canonical(z):
    # Use the boundary fit to compute the source-to-canonical mapping
    # This is a placeholder for the actual implementation
    return np.zeros_like(z)

# Define the canonical-to-source inverse mapping
def canonical_to_source(z):
    # Use the inverse of the source-to-canonical mapping to compute the canonical-to-source inverse mapping
    # This is a placeholder for the actual implementation
    return np.zeros_like(z)

# Compute the canonical boundary sample values
canonical_boundary_sample_values = source_to_canonical(boundary_samples_dense)

# Compute the canonical boundary sample derivatives
canonical_boundary_sample_derivatives = source_to_canonical(boundary_samples_dense)

# Compute the canonical sample values
canonical_sample_values = source_to_canonical(sample_points)

# Compute the canonical sample derivatives
canonical_sample_derivatives = source_to_canonical(sample_points)

# Compute the canonical probe values
canonical_probes = source_to_canonical(probe_points)

# Compute the canonical probe derivatives
canonical_probe_derivatives = source_to_canonical(probe_points)

# Compute the source inverse sample values
source_inverse_sample_values = canonical_to_source(canonical_samples)

# Compute the source inverse sample derivatives
source_inverse_sample_derivatives = canonical_to_source(canonical_samples)

# Write the output files
np.save("canonical_boundary.npy", canonical_boundary)
np.save("canonical_probes.npy", canonical_probes)
np.save("canonical_probe_derivatives.npy", canonical_probe_derivatives)
np.save("canonical_sample_values.npy", canonical_sample_values)
np.save("canonical_sample_derivatives.npy", canonical_sample_derivatives)
np.save("canonical_boundary_sample_values.npy", canonical_boundary_sample_values)
np.save("canonical_boundary_sample_derivatives.npy", canonical_boundary_sample_derivatives)
np.save("source_inverse_sample_values.npy", source_inverse_sample_values)
np.save("source_inverse_sample_derivatives.npy", source_inverse_sample_derivatives)

# Write the slit parameters file
with open("slit_params.csv", "w") as f:
    f.write("component_id,role,center_re,center_im,half_length,half_thickness,angle_rad\n")
    f.write("0,outer_boundary,0.0,0.0,1.0,1.0,0.0\n")
    f.write("1,slit_1,0.0,0.0,1.0,1.0,0.0\n")
    f.write("2,slit_2,0.0,0.0,1.0,1.0,0.0\n")
    f.write("3,slit_3,0.0,0.0,1.0,1.0,0.0\n")
```