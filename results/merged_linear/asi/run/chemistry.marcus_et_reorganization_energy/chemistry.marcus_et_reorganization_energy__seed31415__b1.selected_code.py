import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load data
site_table = pd.read_csv('data/site_table.csv')
frame_site_order = pd.read_csv('data/frame_site_order.csv')
frame_a_coords = np.load('data/frame_a_coords.npy')
frame_b_coords = np.load('data/frame_b_coords.npy')
frame_a_hessian = np.load('data/frame_a_hessian.npy')
frame_b_hessian = np.load('data/frame_b_hessian.npy')
system_manifest = json.load(open('data/system_manifest.json'))
medium_response = json.load(open('data/medium_response.json'))
driving_force_grid = pd.read_csv('data/driving_force_grid.csv')

# Step 2: Reorder coordinates and Hessian by site_id
frame_site_order = frame_site_order.sort_values('site_id').reset_index(drop=True)
site_id_to_idx = dict(frame_site_order[['site_id', 'row_index']].values)
frame_a_coords = frame_a_coords[site_id_to_idx.values]
frame_b_coords = frame_b_coords[site_id_to_idx.values]
frame_a_hessian = frame_a_hessian[site_id_to_idx.values][:, site_id_to_idx.values]
frame_b_hessian = frame_b_hessian[site_id_to_idx.values][:, site_id_to_idx.values]

# Step 4: Forward alignment (frame B to A)
from scipy.optimize import minimize

def align_coordinates(coords_a, coords_b):
    def objective(R, t):
        return np.sum((coords_b @ R + t - coords_a) ** 2)
    R = np.eye(3)
    t = np.zeros(3)
    result = minimize(lambda x: objective(x[:9].reshape(3,3), x[9:]), np.concatenate([R.flatten(), t]))
    return result.x[:9].reshape(3,3), result.x[9:]

R_forward, t_forward = align_coordinates(frame_a_coords, frame_b_coords)
aligned_b = frame_b_coords @ R_forward + t_forward
delta_forward = aligned_b - frame_a_coords

# Step 5: Backward alignment (frame A to B)
R_backward, t_backward = align_coordinates(frame_b_coords, frame_a_coords)
aligned_a = frame_a_coords @ R_backward + t_backward
delta_backward = aligned_a - frame_b_coords

# Step 6: Mass-weighted aligned displacements
masses = site_table['mass_amu'].values
mass_weighted_delta_forward = delta_forward * masses
mass_weighted_delta_backward = delta_backward * masses

np.save('results/aligned_displacement_mw.npy', np.vstack([mass_weighted_delta_forward, mass_weighted_delta_backward]))

# Step 8: Mass-weighted Hessians and eigenvectors
def process_hessian(hessian, masses):
    hessian_mw = hessian * masses[:, np.newaxis] * masses[np.newaxis, :]
    eigenvalues, eigenvectors = np.linalg.eigh(hessian_mw)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors

eigenvalues_a, eigenvectors_a = process_hessian(frame_a_hessian, masses)
eigenvalues_b, eigenvectors_b = process_hessian(frame_b_hessian, masses)

# Step 9: Transport and overlap
def transport_basis(eigenvectors, R):
    return eigenvectors @ R.T

R_forward_transport = np.eye(3)
R_backward_transport = np.eye(3)
transported_basis_forward = transport_basis(eigenvectors_b, R_forward_transport)
transported_basis_backward = transport_basis(eigenvectors_a, R_backward_transport)

overlap_forward = np.abs(transported_basis_forward.T @ eigenvectors_a)
overlap_backward = np.abs(transported_basis_backward.T @ eigenvectors_b)

np.save('results/mode_overlap.npy', np.vstack([overlap_forward, overlap_backward]))

# Step 10: Projected displacements
projected_forward = np.sum(transported_basis_forward * mass_weighted_delta_forward, axis=1)
projected_backward = np.sum(transported_basis_backward * mass_weighted_delta_backward, axis=1)

np.save('results/projected_displacement_mw.npy', np.vstack([projected_forward, projected_backward]))

# Step 11: Inner-sphere four-point terms
lambda_4pt_forward = 0.5 * mass_weighted_delta_forward.T @ frame_a_hessian @ mass_weighted_delta_forward
lambda_4pt_backward = 0.5 * mass_weighted_delta_backward.T @ frame_b_hessian @ mass_weighted_delta_backward

# Step 12: Local resolved-mode terms
eigenvalues_a = eigenvalues_a[::-1]
eigenvalues_b = eigenvalues_b[::-1]
eigenvectors_a = eigenvectors_a[:, ::-1]
eigenvectors_b = eigenvectors_b[:, ::-1]

lambda_nm_forward = np.zeros(15)
lambda_nm_backward = np.zeros(15)

for i in range(15):
    deltaQ_k_forward = projected_forward[i] * eigenvectors_a[i]
    lambda_k_forward = 0.5 * eigenvalues_a[i] * deltaQ_k_forward ** 2
    lambda_nm_forward[i] = lambda_k_forward

    deltaQ_k_backward = projected_backward[i] * eigenvectors_b[i]
    lambda_k_backward = 0.5 * eigenvalues_b[i] * deltaQ_k_backward ** 2
    lambda_nm_backward[i] = lambda_k_backward

# Step 14: Transported profile
transported_forward = transport_basis(eigenvectors_b, R_forward_transport)
transported_backward = transport_basis(eigenvectors_a, R_backward_transport)

abs_deltaQ_transport_forward = np.abs(transported_forward.T @ mass_weighted_delta_forward)
abs_deltaQ_transport_backward = np.abs(transported_backward.T @ mass_weighted_delta_backward)

lambda_transport_forward = 0.5 * eigenvalues_a * abs_deltaQ_transport_forward ** 2
lambda_transport_backward = 0.5 * eigenvalues_b * abs_deltaQ_transport_backward ** 2

np.save('results/transported_mode_projection.npy', np.vstack([abs_deltaQ_transport_forward, lambda_transport_forward]))

# Step 16: Sum local per-mode terms
lambda_nm_forward = np.sum(lambda_nm_forward)
lambda_nm_backward = np.sum(lambda_nm_backward)

# Step 17: Effective inner-sphere term
lambda_inner_effective = np.sqrt(lambda_4pt_forward * lambda_4pt_backward)

# Step 18: Outer-sphere term
lambda_outer = (1 / (2 * medium_response['shell_0_A']) + 1 / (2 * medium_response['shell_1_A']) - 1 / medium_response['span_A']) * (1 / medium_response['eps_fast'] - 1 / medium_response['eps_slow'])

# Step 19: Total lambda
lambda_total = lambda_inner_effective + lambda_outer

# Step 20: Marcus ET rate
hbar = 6.582e-16  # eV·s
k_B = 8.617e-5  # eV/K
T = medium_response['temp_K']
H_ab = medium_response['coupling_eV']

driving_force_grid['k_et_s_inv'] = 0
driving_force_grid['log10_k_et_s_inv'] = 0

for index, row in driving_force_grid.iterrows():
    axis_eV = row['axis_eV']
    numerator = (2 * np.pi / hbar) * H_ab ** 2
    denominator = np.sqrt(4 * np.pi * lambda_total * k_B * T)
    exponent = -(axis_eV + lambda_total) ** 2 / (4 * lambda_total * k_B * T)
    k_et = numerator / denominator * np.exp(exponent)
    driving_force_grid.loc[index, 'k_et_s_inv'] = k_et
    driving_force_grid.loc[index, 'log10_k_et_s_inv'] = np.log10(k_et)

# Step 22: Detect Marcus inverted region
inverted_region_detected = False
first_inverted_grid_id = None
first_inverted_axis_eV = None

for index, row in driving_force_grid.sort_values('axis_eV').iterrows():
    if index > 0 and row['k_et_s_inv'] < driving_force_grid.loc[index - 1, 'k_et_s_inv']:
        inverted_region_detected = True
        first_inverted_grid_id = row['grid_id']
        first_inverted_axis_eV = row['axis_eV']
        break

# Step 21: Self-exchange rate
k_self_exchange = driving_force_grid[driving_force_grid['axis_eV'] == 0]['k_et_s_inv'].values[0]

# Step 13: Mode projection
mode_projection = np.zeros((2, 15, 3))
mode_projection[0, :, 0] = eigenvalues_a
mode_projection[0, :, 1] = projected_forward
mode_projection[0, :, 2] = lambda_nm_forward

mode_projection[1, :, 0] = eigenvalues_b
mode_projection[1, :, 1] = projected_backward
mode_projection[1, :, 2] = lambda_nm_backward

np.save('results/mode_projection.npy', mode_projection)

# Step 23: Summary
lambda_summary = {
    'lambda_4pt_forward_eV': lambda_4pt_forward,
    'lambda_4pt_backward_eV': lambda_4pt_backward,
    'lambda_nm_forward_eV': lambda_nm_forward,
    'lambda_nm_backward_eV': lambda_nm_backward,
    'lambda_inner_effective_eV': lambda_inner_effective,
    'lambda_outer_eV': lambda_outer,
    'lambda_total_eV': lambda_total,
    'k_self_exchange_s_inv': k_self_exchange
}

with open('results/lambda_summary.csv', 'w') as f:
    f.write('lambda_4pt_forward_eV,lambda_4pt_backward_eV,lambda_nm_forward_eV,lambda_nm_backward_eV,lambda_inner_effective_eV,lambda_outer_eV,lambda_total_eV,k_self_exchange_s_inv\n')
    f.write(f"{lambda_summary['lambda_4pt_forward_eV']},{lambda_summary['lambda_4pt_backward_eV']},{lambda_summary['lambda_nm_forward_eV']},{lambda_summary['lambda_nm_backward_eV']},{lambda_summary['lambda_inner_effective_eV']},{lambda_summary['lambda_outer_eV']},{lambda_summary['lambda_total_eV']},{lambda_summary['k_self_exchange_s_inv']}\n")

# Step 24: Rate curve
driving_force_grid.to_csv('results/rate_curve.csv', index=False)

# Step 25: Diagnostics
diagnostics = {
    'alignment_rmsd_forward_A': np.sqrt(np.sum((aligned_b - frame_a_coords) ** 2)),
    'alignment_rmsd_backward_A': np.sqrt(np.sum((aligned_a - frame_b_coords) ** 2)),
    'closure_error_forward_eV': np.abs(lambda_4pt_forward - lambda_nm_forward),
    'closure_error_backward_eV': np.abs(lambda_4pt_backward - lambda_nm_backward),
    'effective_inner_consistency_eV': np.abs(lambda_4pt_forward - lambda_4pt_backward),
    'rigid_norm_forward': np.sqrt(np.sum(R_forward ** 2)),
    'rigid_norm_backward': np.sqrt(np.sum(R_backward ** 2)),
    'projected_norm_forward': np.sqrt(np.sum(projected_forward ** 2)),
    'projected_norm_backward': np.sqrt(np.sum(projected_backward ** 2)),
    'dominant_mode_forward': np.argmax(lambda_nm_forward),
    'dominant_mode_backward': np.argmax(lambda_nm_backward),
    'rate_peak_grid_id': driving_force_grid['grid_id'].iloc[np.argmax(driving_force_grid['k_et_s_inv'])],
    'rate_peak_axis_eV': driving_force_grid['axis_eV'].iloc[np.argmax(driving_force_grid['k_et_s_inv'])],
    'rate_peak_s_inv': driving_force_grid['k_et_s_inv'].iloc[np.argmax(driving_force_grid['k_et_s_inv'])],
    'inverted_region_detected': inverted_region_detected,
    'first_inverted_grid_id': first_inverted_grid_id,
    'first_inverted_axis_eV': first_inverted_axis_eV
}

with open('results/diagnostics.json', 'w') as f:
    json.dump(diagnostics, f, indent=4)

# Step 26: Overview plot
plt.figure(figsize=(12, 6))
plt.plot(driving_force_grid['axis_eV'], driving_force_grid['k_et_s_inv'], label='Rate Curve')
plt.xlabel('Axis (eV)')
plt.ylabel('Rate (s^{-1})')
plt.title('Marcus Electron Transfer Rate Curve')
plt.legend()
plt.savefig('results/overview.png')
plt.close()