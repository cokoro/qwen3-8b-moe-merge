```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load data
site_table = pd.read_csv('data/site_table.csv')
frame_site_order = pd.read_csv('data/frame_site_order.csv')
coords_a = np.load('data/frame_a_coords.npy')
coords_b = np.load('data/frame_b_coords.npy')
hess_a = np.load('data/frame_a_hessian.npy')
hess_b = np.load('data/frame_b_hessian.npy')
manifest = pd.read_json('data/system_manifest.json')
medium = pd.read_json('data/medium_response.json')
grid = pd.read_csv('data/driving_force_grid.csv')

# Step 2: Reorder coordinates and Hessians
frame_order = frame_site_order.sort_values(['frame_id', 'row_index']).set_index('site_id')
coords_a = coords_a[frame_order.index.values]
coords_b = coords_b[frame_order.index.values]
hess_a = hess_a[np.ix_(frame_order.index.values, frame_order.index.values)]
hess_b = hess_b[np.ix_(frame_order.index.values, frame_order.index.values)]

# Step 3: Compute aligned displacements
def align(coords_source, coords_target, mass_weights):
    # Use SVD to find rigid alignment
    pass

# Step 4-5: Compute forward/backward aligned displacements
delta_forward = align(coords_b, coords_a, mass_weights='mass_amu')
delta_backward = align(coords_a, coords_b, mass_weights='mass_amu')

# Step 6: Write aligned displacements
np.save('results/aligned_displacement_mw.npy', np.stack([delta_forward, delta_backward]))

# Step 7: Diagonalize Hessians
def diagonalize_hessian(hessian, mass_weights):
    # Apply mass-weighting and diagonalize
    pass

basis_a = diagonalize_hessian(hess_a, mass_weights='mass_amu')
basis_b = diagonalize_hessian(hess_b, mass_weights='mass_amu')

# Step 8: Write vibrational basis
np.save('results/vibrational_basis_mw.npy', np.stack([basis_a, basis_b]))

# Step 9: Compute mode overlaps
def compute_overlap(basis_source, basis_target, transport_matrix):
    # Compute overlap matrix
    pass

overlap_forward = compute_overlap(basis_b, basis_a, transport_matrix='R.T')
overlap_backward = compute_overlap(basis_a, basis_b, transport_matrix='R.T')
np.save('results/mode_overlap.npy', np.stack([overlap_forward, overlap_backward]))

# Step 10: Project displacements
projected_forward = project_delta(delta_forward, basis_a)
projected_backward = project_delta(delta_backward, basis_b)
np.save('results/projected_displacement_mw.npy', np.stack([projected_forward, projected_backward]))

# Step 11-13: Compute lambda_4pt and mode projections
lambda_4pt_forward = 0.5 * delta_forward.T @ hess_a @ delta_forward
lambda_4pt_backward = 0.5 * delta_backward.T @ hess_b @ delta_backward

mode_projection = []
for k in range(15):
    freq = freq_scale_factor * sqrt(omega_k_squared)
    deltaQ = basis_k.T @ projected_forward
    lambda_k = 0.5 * omega_k_squared * deltaQ**2
    mode_projection.append([freq, deltaQ, lambda_k])
np.save('results/mode_projection.npy', np.stack(mode_projection))

# Step 14: Transport projections
transported_forward = transport_basis(basis_b, basis_a)
transported_backward = transport_basis(basis_a, basis_b)
np.save('results/transported_mode_projection.npy', np.stack([transported_forward, transported_backward]))

# Step 15: Compute lambda_total and ET rate
lambda_total = lambda_inner_effective + lambda_outer
grid['k_et_s_inv'] = compute_rate(grid['axis_eV'], lambda_total)
grid['log10_k_et_s_inv'] = np.log10(grid['k_et_s_inv'])
grid.to_csv('results/rate_curve.csv', index=False)

# Step 16: Write lambda_summary
lambda_summary = {
    'lambda_4pt_forward_eV': lambda_4pt_forward,
    'lambda_4pt_backward_eV': lambda_4pt_backward,
    'lambda_nm_forward_eV': lambda_nm_forward,
    'lambda_nm_backward_eV': lambda_nm_backward,
    'lambda_inner_effective_eV': lambda_inner_effective,
    'lambda_outer_eV': lambda_outer,
    'lambda_total_eV': lambda_total,
    'k_self_exchange_s_inv': k_self_exchange,
}
pd.DataFrame([lambda_summary]).to_csv('results/lambda_summary.csv', index=False)

# Step 17: Write diagnostics
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
    'first_inverted_axis_eV': first_inverted_axis_eV,
}
with open('results/diagnostics.json', 'w') as f:
    json.dump(diagnostics, f)

# Step 18: Plot overview
plt.figure(figsize=(10, 6))
plt.plot(grid['axis_eV'], grid['k_et_s_inv'])
plt.xlabel('Axis (eV)')
plt.ylabel('Rate (s^-1)')
plt.title('Marcus ET Rate Curve')
plt.savefig('results/overview.png')
```