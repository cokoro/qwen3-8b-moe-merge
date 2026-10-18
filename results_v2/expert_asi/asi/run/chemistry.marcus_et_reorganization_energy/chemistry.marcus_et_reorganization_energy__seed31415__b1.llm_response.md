```python
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.linalg as la

# Step 1: Read masses from site_table.csv
site_table = pd.read_csv('data/site_table.csv')
masses = site_table['mass_amu'].values

# Step 2: Read frame_site_order.csv and reorder coordinates and Hessians
frame_site_order = pd.read_csv('data/frame_site_order.csv')
site_id_order = frame_site_order['site_id'].values

# Load frame_a and frame_b coordinates and Hessians
frame_a_coords = np.load('data/frame_a_coords.npy')
frame_b_coords = np.load('data/frame_b_coords.npy')
frame_a_hessian = np.load('data/frame_a_hessian.npy')
frame_b_hessian = np.load('data/frame_b_hessian.npy')

# Reorder each frame's coordinates and Hessian by site_id
frame_a_coords = frame_a_coords[site_id_order]
frame_b_coords = frame_b_coords[site_id_order]

frame_a_hessian = frame_a_hessian[site_id_order]
frame_a_hessian = frame_a_hessian[:, site_id_order]
frame_b_hessian = frame_b_hessian[site_id_order]
frame_b_hessian = frame_b_hessian[:, site_id_order]

# Step 3: Define forward and backward structural directions
# Step 4: Align frame B onto frame A
# Step 5: Align frame A onto frame B
# Step 6: Compute aligned displacements
delta_forward = frame_b_coords @ R_forward - frame_a_coords
delta_backward = frame_a_coords @ R_backward - frame_b_coords

# Step 7: Write aligned displacements to results/aligned_displacement_mw.npy
aligned_displacement = np.stack([delta_forward, delta_backward], axis=0)
np.save('results/aligned_displacement_mw.npy', aligned_displacement)

# Step 8: Build mass-weighted Hessians
frame_a_hessian_mw = frame_a_hessian * (masses[None, :] / 1e3)  # Convert amu to kg
frame_b_hessian_mw = frame_b_hessian * (masses[None, :] / 1e3)

# Diagonalize Hessians and discard null_rank modes
frame_a_eigenvalues, frame_a_eigenvectors = la.eigh(frame_a_hessian_mw)
frame_b_eigenvalues, frame_b_eigenvectors = la.eigh(frame_b_hessian_mw)

# Keep only retained modes
retained_modes = 21 - null_rank
frame_a_basis = frame_a_eigenvectors[:, :retained_modes]
frame_b_basis = frame_b_eigenvectors[:, :retained_modes]

# Step 9: Transport basis and compute overlaps
transported_basis_forward = frame_a_basis @ R_forward
transported_basis_backward = frame_b_basis @ R_backward

overlap_forward = np.abs(transported_basis_forward @ transported_basis_backward.T)
overlap_backward = np.abs(transported_basis_backward @ transported_basis_forward.T)

# Step 10: Project displacements onto retained modes
projected_displacement = np.stack([delta_forward, delta_backward], axis=0)
np.save('results/projected_displacement_mw.npy', projected_displacement)

# Step 11: Compute inner-sphere four-point terms
lambda_4pt_forward = 0.5 * delta_forward.T @ frame_a_hessian_mw @ delta_forward
lambda_4pt_backward = 0.5 * delta_backward.T @ frame_b_hessian_mw @ delta_backward

# Step 12: Compute local per-mode terms
mode_projection = np.zeros((2, retained_modes, 3))
for idx in range(retained_modes):
    v_k = frame_a_basis[:, idx] if idx < retained_modes else frame_b_basis[:, idx]
    deltaQ_k = v_k.T @ projected_displacement[0]
    omega_k = np.sqrt(frame_a_eigenvalues[idx] if idx < retained_modes else frame_b_eigenvalues[idx])
    lambda_mode_eV = 0.5 * omega_k**2 * deltaQ_k**2
    freq_cm_inv = freq_scale_factor * omega_k
    mode_projection[0, idx] = [freq_cm_inv, deltaQ_k, lambda_mode_eV]
    mode_projection[1, idx] = [freq_cm_inv, deltaQ_k, lambda_mode_eV]

np.save('results/mode_projection.npy', mode_projection)

# Step 13: Transport profiles
transported_projection = np.zeros((2, retained_modes, 2))
for idx in range(retained_modes):
    transport = transported_basis_forward[:, idx] if idx < retained_modes else transported_basis_backward[:, idx]
    abs_deltaQ = np.abs(transport.T @ projected_displacement[0])
    lambda_transport = 0.5 * frame_a_eigenvalues[idx] * abs_deltaQ**2
    transported_projection[0, idx] = [abs_deltaQ, lambda_transport]
    transported_projection[1, idx] = [abs_deltaQ, lambda_transport]

np.save('results/transported_mode_projection.npy', transported_projection)

# Step 14: Sum local per-mode terms
lambda_nm_forward = np.sum(mode_projection[0, :, 2])
lambda_nm_backward = np.sum(mode_projection[1, :, 2])

# Step 15: Compute effective inner-sphere term
lambda_inner_effective = np.sqrt(lambda_4pt_forward * lambda_4pt_backward)

# Step 16: Compute outer-sphere term
lambda_outer = (1 / (2 * shell_0_A) + 1 / (2 * shell_1_A) - 1 / span_A) * (1 / eps_fast - 1 / eps_slow)

# Step 17: Total lambda
lambda_total = lambda_inner_effective + lambda_outer

# Step 18: Compute rate curve
driving_force_grid = pd.read_csv('data/driving_force_grid.csv')
grid_ids = driving_force_grid['grid_id'].values
axis_eV = driving_force_grid['axis_eV'].values

# Step 19: Compute k_ET
k_ET = (2 * np.pi / hbar) * np.abs(H_ab)**2 / np.sqrt(4 * np.pi * lambda_total * k_B * T) * np.exp(- (axis_eV + lambda_total)**2 / (4 * lambda_total * k_B * T))

# Step 20: Write rate curve
rate_curve = pd.DataFrame({
    'grid_id': grid_ids,
    'axis_eV': axis_eV,
    'k_et_s_inv': k_ET,
    'log10_k_et_s_inv': np.log10(k_ET)
})
rate_curve.to_csv('results/rate_curve.csv', index=False)

# Step 21: Self-exchange rate at axis_eV=0
k_self_exchange = k_ET[np.abs(axis_eV) < 1e-6]

# Step 22: Detect inverted region
inverted_region_detected = False
first_inverted_grid_id = None
first_inverted_axis_eV = None

# Step 23: Write lambda_summary.csv
lambda_summary = pd.DataFrame({
    'lambda_4pt_forward_eV': [lambda_4pt_forward],
    'lambda_4pt_backward_eV': [lambda_4pt_backward],
    'lambda_nm_forward_eV': [lambda_nm_forward],
    'lambda_nm_backward_eV': [lambda_nm_backward],
    'lambda_inner_effective_eV': [lambda_inner_effective],
    'lambda_outer_eV': [lambda_outer],
    'lambda_total_eV': [lambda_total],
    'k_self_exchange_s_inv': [k_self_exchange]
})
lambda_summary.to_csv('results/lambda_summary.csv', index=False)

# Step 24: Write diagnostics.json
diagnostics = {
    'alignment_rmsd_forward_A': rmsd_forward,
    'alignment_rmsd_backward_A': rmsd_backward,
    'closure_error_forward_eV': closure_error_forward,
    'closure_error_backward_eV': closure_error_backward,
    'effective_inner_consistency_eV': effective_inner_consistency,
    'rigid_norm_forward': rigid_norm_forward,
    'rigid_norm_backward': rigid_norm_backward,
    'projected_norm_forward': projected_norm_forward,
    'projected_norm_backward': projected_norm_backward,
    'dominant_mode_forward': dominant_mode_forward,
    'dominant_mode_backward': dominant_mode_backward,
    'rate_peak_grid_id': rate_peak_grid_id,
    'rate_peak_axis_eV': rate_peak_axis_eV,
    'rate_peak_s_inv': rate_peak_s_inv,
    'inverted_region_detected': inverted_region_detected,
    'first_inverted_grid_id': first_inverted_grid_id,
    'first_inverted_axis_eV': first_inverted_axis_eV
}
with open('results/diagnostics.json', 'w') as f:
    json.dump(diagnostics, f)

# Step 25: Plot rate curve and write overview.png
plt.figure(figsize=(8, 6))
plt.plot(axis_eV, np.log10(k_ET), 'b-')
plt.xlabel('Axis (eV)')
plt.ylabel('log10(k_ET/s^-1)')
plt.savefig('results/overview.png')
plt.close()
```