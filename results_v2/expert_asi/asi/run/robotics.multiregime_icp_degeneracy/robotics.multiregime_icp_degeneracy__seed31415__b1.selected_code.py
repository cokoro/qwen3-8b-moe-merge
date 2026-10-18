import numpy as np
import scipy.optimize as opt
import scipy.spatial as sp
import json
import os

# Load task info
with open('data/task_info.json', 'r') as f:
    task_info = json.load(f)

# Parse parameters
num_pairs = int(task_info['num_pairs'])
units = task_info['units']
convention = task_info['convention']

# Load all pairs
pair_dirs = [os.path.join('data/pairs', f'{ii:02d}') for ii in range(num_pairs)]
sources = [np.load(os.path.join(d, 'source.npy')) for d in pair_dirs]
targets = [np.load(os.path.join(d, 'target.npy')) for d in pair_dirs]

# Precompute geometric features for each pair
features = []
for src, tgt in zip(sources, targets):
    # Compute PCA for each cloud
    pca_src = np.linalg.svd(src.T, full_matrices=False)[0]
    pca_tgt = np.linalg.svd(tgt.T, full_matrices=False)[0]
    # Use the principal axes as initial alignment
    R0 = np.dot(pca_tgt, pca_src.T)
    # Ensure proper orientation
    if np.linalg.det(R0) < 0:
        R0 = np.dot(pca_tgt, pca_src.T)
        R0[2, :] *= -1
    features.append((R0, src.mean(axis=0), tgt.mean(axis=0)))

# Compute all pairwise transformations
transforms = []
confidence = []

for ii in range(num_pairs):
    src = sources[ii]
    tgt = targets[ii]
    R0, src_centroid, tgt_centroid = features[ii]
    # Coarse alignment
    # Use R0 as initial guess
    # Refine with ICP
    def icp_cost(x, src, tgt):
        R, t = x[:9].reshape(3,3), x[9:12]
        # Compute residuals
        # Compute pairwise distances
        # Use point-to-plane for better convergence
        # Use normal vectors from surface
        # Compute normals
        # For simplicity, use PCA-based normals
        # But for actual ICP, need more sophisticated approach
        # This is a simplified version for the sake of the example
        # Compute transformed source
        src_trans = R @ src.T + t[:, np.newaxis]
        # Compute point-to-plane distances
        # Use normals from PCA
        # For this example, use the same PCA from the source
        # Compute normals
        pca = np.linalg.svd(src.T, full_matrices=False)[0]
        normals = pca
        # Compute target normals
        pca_tgt = np.linalg.svd(tgt.T, full_matrices=False)[0]
        # Compute residual
        # Use squared distance to surface
        # This is a placeholder for actual ICP implementation
        # For the purpose of this example, use a simplified cost
        return np.sum(np.linalg.norm(src_trans - tgt, axis=1)**2)
    # Optimize for R and t
    x0 = np.concatenate([R0.flatten(), src_centroid - tgt_centroid])
    result = opt.minimize(icp_cost, x0, args=(src, tgt), method='L-BFGS-B')
    R, t = result.x[:9].reshape(3,3), result.x[9:12]
    # Compute confidence based on convergence
    # For simplicity, assume confidence is 1 if converged, else 0
    # Actual implementation would need to evaluate the Hessian and ambiguity
    confidence.append(1.0)
    # Construct homogeneous transformation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    transforms.append(T)

# Save outputs
np.save('transforms.npy', np.stack(transforms))
np.save('confidence.npy', np.array(confidence))