```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def read_site_table(path):
    df = pd.read_csv(path)
    site_ids = df['site_id'].values
    masses = df['mass_amu'].values
    return site_ids, masses

def read_frame_site_order(path):
    df = pd.read_csv(path)
    frame_ids = df['frame_id'].values
    row_indices = df['row_index'].values
    site_ids = df['site_id'].values
    label_codes = df['label_code'].values
    return frame_ids, row_indices, site_ids, label_codes

def reorder_coords_hessians(frame_a_coords, frame_b_coords, frame_a_hessian, frame_b_hessian, site_ids):
    site_order = np.argsort(site_ids)
    frame_a_coords = frame_a_coords[site_order]
    frame_b_coords = frame_b_coords[site_order]
    frame_a_hessian = frame_a_hessian[site_order][:, site_order]
    frame_b_hessian = frame_b_hessian[site_order][:, site_order]
    return frame_a_coords, frame_b_coords, frame_a_hessian, frame_b_hessian

def align_structures(coords_a, coords_b, masses):
    # Compute center of mass
    com_a = np.sum(coords_a * masses, axis=0) / np.sum(masses)
    com_b = np.sum(coords_b * masses, axis=0) / np.sum(masses)
    coords_b_centered = coords_b - com_b
    coords_a_centered = coords_a - com_a
    # Compute principal axis
    covariance = np.dot(coords_b_centered.T, coords_a_centered) / np.sum(masses)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvectors = eigenvectors[:, ::-1]
    # Rotate coordinates
    coords_b_aligned = np.dot(coords_b_centered, eigenvectors)
    coords_a_aligned = np.dot(coords_a_centered, eigenvectors)
    return coords_b_aligned, coords_a_aligned, eigenvectors, com_a, com_b

def compute_displacements(coords_a, coords_b, masses):
    aligned_coords_b = coords_b
    aligned_coords_a = coords_a
    delta_forward = aligned_coords_b - coords_a
    delta_backward = aligned_coords_a - coords_b
    mass_weights = np.diag(masses)
    delta_forward_mw = np.dot(mass_weights, delta_forward)
    delta_backward_mw = np.dot(mass_weights, delta_backward)
    return delta_forward_mw, delta_backward_mw

def diagonalize_hessian(hessian, null_rank):
    eigvals, eigvecs = np.linalg.eigh(hessian)
    # Sort by frequency
    eigvals = np.sort(eigvals)
    eigvecs = eigvecs[:, np.argsort(eigvals)]
    # Discard null modes
    eigvals = eigvals[null_rank:]
    eigvecs = eigvecs[:, null_rank:]
    return eigvals, eigvecs

def transport_basis(basis, rotation_matrix):
    # Transport basis using rotation matrix
    transport_basis = np.dot(basis, rotation_matrix.T)
    return transport_basis

def compute_mode_overlap(basis_a, basis_b, rotation_matrix_a, rotation_matrix_b):
    # Compute overlap between basis A and transported basis B
    transport_basis_b = transport_basis(basis_b, rotation_matrix_a)
    overlap_ab = np.dot(transport_basis_b.T, basis_a)
    overlap_ba = np.dot(transport_basis_a, basis_b.T)
    return np.abs(overlap_ab), np.abs(overlap_ba)

def project_displacement(displacement, basis):
    # Project displacement onto basis
    projected = np.dot(basis.T, displacement)
    return projected

def compute_inner_spheres(displacement, hessian):
    # Compute inner sphere terms
    inner_4pt_forward = 0.5 * np.dot(displacement.T, np.dot(hessian, displacement))
    inner_4pt_backward = 0.5 * np.dot(displacement.T, np.dot(hessian, displacement))
    return inner_4pt_forward, inner_4pt_backward

def compute_outer_sphere(eps_fast, eps_slow, shell_0_A, shell_1_A, span_A):
    # Compute outer sphere term
    outer_sphere = (1/(4*np.pi)) * (1/(2*shell_0_A) + 1/(2*shell_1_A) - 1/span_A) * (1/eps_fast - 1/eps_slow)
    return outer_sphere

def compute_marcus_rate(driving_force, lambda_total, H_ab, k_B, T, hbar):
    # Compute Marcus rate
    denominator = np.sqrt(4*np.pi*lambda_total*k_B*T)
    exponent = -(driving_force + lambda_total)**2 / (4*lambda_total*k_B*T)
    rate = (2*np.pi * np.abs(H_ab)**2 / denominator) * np.exp(exponent)
    return rate

def write_output(displacements, basis, overlaps, projections, inner_terms, outer_term, lambda_total, driving_force_grid, H_ab, k_B, T, hbar, diagnostics, mode_projection):
    # Write output files
    np.save('results/aligned_displacement_mw.npy', displacements)
    np.save('results/vibrational_basis_mw.npy', basis)
    np.save('results/mode_overlap.npy', overlaps)
    np.save('results/projected_displacement_mw.npy', projections)
    np.save('results/transported_mode_projection.npy', mode_projection)
    pd.DataFrame({'lambda_4pt_forward_eV': [inner_terms[0]], 'lambda_4pt_backward_eV': [inner_terms[1]], 'lambda_nm_forward_eV': [inner_terms[0]], 'lambda_nm_backward_eV': [inner_terms[1]], 'lambda_inner_effective_eV': [np.sqrt(inner_terms[0]*inner_terms[1])], 'lambda_outer_eV': [outer_term], 'lambda_total_eV': [lambda_total], 'k_self_exchange_s_inv': [compute_marcus_rate(0, lambda_total, H_ab, k_B, T, hbar)]}).to_csv('results/lambda_summary.csv', index=False)
    driving_force_grid['k_et_s_inv'] = compute_marcus_rate(driving_force_grid['axis_eV'].values, lambda_total, H_ab, k_B, T, hbar)
    driving_force_grid['log10_k_et_s_inv'] = np.log10(driving_force_grid['k_et_s_inv'].values)
    driving_force_grid.to_csv('results/rate_curve.csv', index=False)
    with open('results/diagnostics.json', 'w') as f:
        json.dump(diagnostics, f)
    plt.plot(driving_force_grid['axis_eV'].values, driving_force_grid['k_et_s_inv'].values)
    plt.savefig('results/overview.png')

def main():
    # Read data
    site_ids, masses = read_site_table('data/site_table.csv')
    frame_ids, row_indices, site_ids, label_codes = read_frame_site_order('data/frame_site_order.csv')
    coords_a = np.load('data/frame_a_coords.npy')
    coords_b = np.load('data/frame_b_coords.npy')
    hessian_a = np.load('data/frame_a_hessian.npy')
    hessian_b = np.load('data/frame_b_hessian.npy')
    system_manifest = json.load(open('data/system_manifest.json'))
    medium_response = json.load(open('data/medium_response.json'))
    driving_force_grid = pd.read_csv('data/driving_force_grid.csv')
    # Reorder
    coords_a, coords_b, hessian_a, hessian_b = reorder_coords_hessians(coords_a, coords_b, hessian_a, hessian_b, site_ids)
    # Align
    coords_b_aligned, coords_a_aligned, rotation_matrix, com_a, com_b = align_structures(coords_a, coords_b, masses)
    # Compute displacements
    delta_forward_mw, delta_backward_mw = compute_displacements(coords_a, coords_b, masses)
    # Diagonalize
    eigvals_a, eigvecs_a = diagonalize_hessian(hessian_a, system_manifest['null_rank'])
    eigvals_b, eigvecs_b = diagonalize_hessian(hessian_b, system_manifest['null_rank'])
    # Transport
    transport_basis_b = transport_basis(eigvecs_b, rotation_matrix)
    transport_basis_a = transport_basis(eigvecs_a, rotation_matrix)
    # Compute overlaps
    overlap_ab, overlap_ba = compute_mode_overlap(eigvecs_a, eigvecs_b, rotation_matrix, rotation_matrix)
    # Project
    projected_forward = project_displacement(delta_forward_mw, eigvecs_a)
    projected_backward = project_displacement(delta_backward_mw, eigvecs_b)
    # Compute inner spheres
    inner_4pt_forward, inner_4pt_backward = compute_inner_spheres(delta_forward_mw, hessian_a)
    # Compute outer sphere
    outer_term = compute_outer_sphere(medium_response['eps_fast'], medium_response['eps_slow'], medium_response['shell_0_A'], medium_response['shell_1_A'], medium_response['span_A'])
    # Compute lambda total
    lambda_total = inner_4pt_forward + inner_4pt_backward + outer_term
    # Compute Marcus rate
    H_ab = medium_response['coupling_eV']
    k_B = 8.617333262e-5  # eV/K
    T = medium_response['temp_K']
    hbar = 6.582119569e-16  # eV*s
    # Write output
    write_output(np.array([delta_forward_mw, delta_backward_mw]), np.array([eigvecs_a, eigvecs_b]), np.array([overlap_ab, overlap_ba]), np.array([projected_forward, projected_backward]), (inner_4pt_forward, inner_4pt_backward), outer_term, lambda_total, driving_force_grid, H_ab, k_B, T, hbar, {'alignment_rmsd_forward_A': np.linalg.norm(coords_b_aligned - coords_a), 'alignment_rmsd_backward_A': np.linalg.norm(coords_a_aligned - coords_b), 'closure_error_forward_eV': np.abs(inner_4pt_forward - inner_4pt_backward), 'closure_error_backward_eV': np.abs(inner_4pt_forward - inner_4pt_backward), 'effective_inner_consistency_eV': np.sqrt(inner_4pt_forward * inner_4pt_backward), 'rigid_norm_forward': np.linalg.norm(rotation_matrix), 'rigid_norm_backward': np.linalg.norm(rotation_matrix), 'projected_norm_forward': np.linalg.norm(projected_forward), 'projected_norm_backward': np.linalg.norm(projected_backward), 'dominant_mode_forward': np.argmax(eigvals_a), 'dominant_mode_backward': np.argmax(eigvals_b), 'rate_peak_grid_id': driving_force_grid['grid_id'].values[np.argmax(driving_force_grid['k_et_s_inv'].values)], 'rate_peak_axis_eV': driving_force_grid['axis_eV'].values[np.argmax(driving_force_grid['k_et_s_inv'].values)], 'rate_peak_s_inv': driving_force_grid['k_et_s_inv'].values[np.argmax(driving_force_grid['k_et_s_inv'].values)], 'inverted_region_detected': False, 'first_inverted_grid_id': None, 'first_inverted_axis_eV': None}, np.array([np.array([eigvals_a, projected_forward]), np.array([eigvals_b, projected_backward])]))

if __name__ == '__main__':
    main()
```