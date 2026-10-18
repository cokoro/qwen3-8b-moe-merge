import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.signal import find_peaks
from scipy.constants import hbar, k, e

# Read data
site_table = pd.read_csv('data/site_table.csv')
frame_site_order = pd.read_csv('data/frame_site_order.csv')
coords_a = np.load('data/frame_a_coords.npy')
coords_b = np.load('data/frame_b_coords.npy')
hess_a = np.load('data/frame_a_hessian.npy')
hess_b = np.load('data/frame_b_hessian.npy')
manifest = json.load(open('data/system_manifest.json'))
medium = json.load(open('data/medium_response.json'))
grid = pd.read_csv('data/driving_force_grid.csv')

# Step 2: Reorder sites
frame_a_order = frame_site_order[frame_site_order['frame_id'] == 'frame_a']['site_id'].values
frame_b_order = frame_site_order[frame_site_order['frame_id'] == 'frame_b']['site_id'].values

# Step 4: Align frame B to frame A
def align_coords(coords, target, mass, verbose=False):
    # Compute center of mass
    mass = mass.reshape(-1,1)
    com = np.sum(coords * mass, axis=0) / np.sum(mass)
    coords -= com
    target -= com
    # Compute rotation matrix
    coords = coords.T
    target = target.T
    # QR decomposition
    q, r = qr(coords.T @ target)
    # Compute RMSD
    rmsd = np.sqrt(np.sum((coords @ q - target)**2) / np.sum(mass))
    # Apply rotation
    aligned = coords @ q
    return aligned + com, q, rmsd

masses = site_table['mass_amu'].values
masses = masses.reshape(-1,1) * 1.66054e-27  # kg
aligned_b_a, rot_b_a, rmsd_b_a = align_coords(coords_b, coords_a, masses)
aligned_a_b, rot_a_b, rmsd_a_b = align_coords(coords_a, coords_b, masses)

# Step 6: Compute displacements
delta_forward = aligned_b_a - coords_a
delta_backward = aligned_a_b - coords_b
delta_forward_mw = delta_forward * masses
delta_backward_mw = delta_backward * masses

# Step 8: Build mass-weighted Hessians
def build_hessian(hess, mass):
    hess_mw = hess * mass
    return hess_mw

hess_a_mw = build_hessian(hess_a, masses)
hess_b_mw = build_hessian(hess_b, masses)

# Diagonalize Hessian and retain modes
def diagonalize_hessian(hess, mass, null_rank):
    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(hess)
    # Sort by frequency
    indices = np.argsort(eigvals)
    eigvals = eigvals[indices]
    eigvecs = eigvecs[:, indices]
    # Remove null-space modes
    eigvals = eigvals[null_rank:]
    eigvecs = eigvecs[:, null_rank:]
    return eigvals, eigvecs

eigvals_a, eigvecs_a = diagonalize_hessian(hess_a_mw, masses, manifest['null_rank'])
eigvals_b, eigvecs_b = diagonalize_hessian(hess_b_mw, masses, manifest['null_rank'])

# Step 9: Transport basis and compute overlaps
def transport_basis(basis, rot):
    # Transport basis to new frame
    transport = basis @ rot
    return transport

basis_a = transport_basis(eigvecs_a, rot_b_a)
basis_b = transport_basis(eigvecs_b, rot_a_b)

# Compute overlaps
def compute_overlap(basis1, basis2):
    # Compute absolute overlaps
    overlap = np.abs(basis1 @ basis2.T)
    return overlap

overlap_forward = compute_overlap(basis_a, eigvecs_b)
overlap_backward = compute_overlap(basis_b, eigvecs_a)

# Step 10: Project displacements
def project_displacement(displacement, basis):
    # Project displacement onto basis
    projected = basis.T @ displacement
    return projected

projected_forward = project_displacement(delta_forward_mw, eigvecs_a)
projected_backward = project_displacement(delta_backward_mw, eigvecs_b)

# Step 11: Compute inner-sphere terms
lambda_4pt_forward = 0.5 * delta_forward_mw.T @ hess_a_mw @ delta_forward_mw
lambda_4pt_backward = 0.5 * delta_backward_mw.T @ hess_b_mw @ delta_backward_mw

# Step 12: Compute local resolved-mode terms
def compute_local_terms(eigvals, basis, projected):
    # Compute frequency in cm^-1
    freq_cm_inv = manifest['freq_scale_factor'] * np.sqrt(eigvals)
    # Compute deltaQ
    deltaQ = projected
    # Compute lambda mode
    lambda_mode = 0.5 * eigvals * deltaQ**2
    return freq_cm_inv, deltaQ, lambda_mode

freq_cm_inv_a, deltaQ_a, lambda_a = compute_local_terms(eigvals_a, eigvecs_a, projected_forward)
freq_cm_inv_b, deltaQ_b, lambda_b = compute_local_terms(eigvals_b, eigvecs_b, projected_backward)

# Step 14: Compute transported terms
transport_forward = basis_a.T @ projected_forward
transport_backward = basis_b.T @ projected_backward
lambda_transport_forward = 0.5 * eigvals_b * transport_forward**2
lambda_transport_backward = 0.5 * eigvals_a * transport_backward**2

# Step 16: Sum local terms
lambda_nm_forward = np.sum(lambda_a)
lambda_nm_backward = np.sum(lambda_b)

# Step 17: Compute effective inner term
lambda_inner_effective = np.sqrt(lambda_4pt_forward * lambda_4pt_backward)

# Step 18: Compute outer-sphere term
lambda_outer = (e**2 / (4 * np.pi * 8.8541878125e-12)) * (1/(2*medium['shell_0_A']) + 1/(2*medium['shell_1_A']) - 1/medium['span_A']) * (1/medium['eps_fast'] - 1/medium['eps_slow'])

# Step 19: Compute total lambda
lambda_total = lambda_inner_effective + lambda_outer

# Step 20: Compute rate curve
def compute_rate(axis, lambda_total):
    # Compute rate
    k_et = (2 * np.pi / hbar) * medium['coupling_eV']**2 / np.sqrt(4 * np.pi * lambda_total * k * medium['temp_K']) * np.exp(-(axis + lambda_total)**2 / (4 * lambda_total * k * medium['temp_K']))
    return k_et

grid['k_et_s_inv'] = grid['axis_eV'].apply(lambda x: compute_rate(x, lambda_total))
grid['log10_k_et_s_inv'] = np.log10(grid['k_et_s_inv'])

# Step 22: Detect inverted region
inverted_region = False
first_inverted = None
for i, row in grid.iterrows():
    if i > 0 and row['k_et_s_inv'] < grid.iloc[i-1]['k_et_s_inv']:
        inverted_region = True
        first_inverted = (row['grid_id'], row['axis_eV'])
        break

# Step 21: Compute self-exchange rate
k_self_exchange = compute_rate(0, lambda_total)

# Step 23: Write outputs
np.save('results/aligned_displacement_mw.npy', np.array([delta_forward_mw, delta_backward_mw]))
np.save('results/vibrational_basis_mw.npy', np.array([eigvecs_a, eigvecs_b]))
np.save('results/mode_overlap.npy', np.array([overlap_forward, overlap_backward]))
np.save('results/projected_displacement_mw.npy', np.array([projected_forward, projected_backward]))
np.save('results/transported_mode_projection.npy', np.array([np.abs(transport_forward), lambda_transport_forward, np.abs(transport_backward), lambda_transport_backward]))
np.save('results/mode_projection.npy', np.array([np.column_stack((freq_cm_inv_a, deltaQ_a, lambda_a)), np.column_stack((freq_cm_inv_b, deltaQ_b, lambda_b))]))
grid.to_csv('results/rate_curve.csv', index=False)
np.savetxt('results/lambda_summary.csv', np.array([lambda_4pt_forward, lambda_4pt_backward, lambda_nm_forward, lambda_nm_backward, lambda_inner_effective, lambda_outer, lambda_total, k_self_exchange]), delimiter=',')
np.save('results/diagnostics.json', json.dumps({
    'alignment_rmsd_forward_A': rmsd_b_a,
    'alignment_rmsd_backward_A': rmsd_a_b,
    'closure_error_forward_eV': lambda_4pt_forward - lambda_nm_forward,
    'closure_error_backward_eV': lambda_4pt_backward - lambda_nm_backward,
    'effective_inner_consistency_eV': lambda_inner_effective,
    'rigid_norm_forward': np.linalg.norm(rot_b_a),
    'rigid_norm_backward': np.linalg.norm(rot_a_b),
    'projected_norm_forward': np.linalg.norm(projected_forward),
    'projected_norm_backward': np.linalg.norm(projected_backward),
    'dominant_mode_forward': np.argmax(np.abs(lambda_a)),
    'dominant_mode_backward': np.argmax(np.abs(lambda_b)),
    'rate_peak_grid_id': grid.loc[grid['k_et_s_inv'].idxmax()]['grid_id'],
    'rate_peak_axis_eV': grid.loc[grid['k_et_s_inv'].idxmax()]['axis_eV'],
    'rate_peak_s_inv': grid.loc[grid['k_et_s_inv'].idxmax()]['k_et_s_inv'],
    'inverted_region_detected': inverted_region,
    'first_inverted_grid_id': first_inverted[0] if first_inverted else None,
    'first_inverted_axis_eV': first_inverted[1] if first_inverted else None
}))
plt.figure(figsize=(10, 6))
plt.plot(grid['axis_eV'], grid['k_et_s_inv'])
plt.xlabel('Axis (eV)')
plt.ylabel('Rate (s^-1)')
plt.savefig('results/overview.png')
plt.close()