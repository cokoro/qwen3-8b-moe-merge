```python
import numpy as np
import json
import os
import scipy.optimize as optimize
import scipy.linalg as linalg
from scipy.spatial import KDTree

def read_pair(ii):
    source = np.load(f"data/pairs/{ii:02d}/source.npy")
    target = np.load(f"data/pairs/{ii:02d}/target.npy")
    return source, target

def pca_alignment(source, target):
    # Compute centroids
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # Center points
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Compute covariance matrix
    cov = source_centered.T @ target_centered
    
    # SVD to find optimal rotation
    u, s, vh = np.linalg.svd(cov)
    r = u @ vh
    
    # Ensure rotation is proper (determinant == 1)
    if np.linalg.det(r) < 0:
        vh[:, -1] *= -1
        r = u @ vh
    
    # Translation vector
    t = target_centroid - r @ source_centroid
    
    return r, t

def icp(source, target, max_iter=50, tolerance=1e-6):
    # Initial guess
    r, t = pca_alignment(source, target)
    source_aligned = r @ source.T + t[:, np.newaxis]
    
    for _ in range(max_iter):
        # Find closest points
        tree = KDTree(target)
        _, indices = tree.query(source_aligned.T)
        
        # Compute cost function
        source_closest = source_aligned.T[indices]
        diff = source_closest - target
        cost = np.sum(diff ** 2) / len(diff)
        
        if cost < tolerance:
            break
        
        # Compute optimal rotation and translation
        m = source_closest.T @ diff
        u, s, vh = np.linalg.svd(m)
        r = u @ vh
        t = np.mean(diff - m @ u, axis=0)
        source_aligned = r @ source.T + t[:, np.newaxis]
    
    return r, t

def compute_hessian(source, target, r, t):
    # Compute residuals
    source_aligned = r @ source.T + t[:, np.newaxis]
    diff = source_aligned.T - target
    
    # Compute Jacobian
    J = np.zeros((len(diff), 6))
    J[:, :3] = -np.kron(np.eye(3), source.T)  # dR/dθ
    J[:, 3:] = -np.kron(np.eye(3), np.eye(3))  # dt/dt
    
    # Compute Hessian
    H = J.T @ J
    
    return H

def get_transform_matrix(r, t):
    return np.block([
        [r, t[:, np.newaxis]],
        [np.zeros((1, 4))]
    ])

def main():
    with open("data/task_info.json", "r") as f:
        task_info = json.load(f)
    
    num_pairs = task_info["num_pairs"]
    transforms = np.zeros((num_pairs, 4, 4), dtype=np.float64)
    confidence = np.zeros(num_pairs, dtype=np.float64)
    
    for ii in range(num_pairs):
        source, target = read_pair(ii)
        r, t = icp(source, target)
        H = compute_hessian(source, target, r, t)
        eigenvalues = np.linalg.eigvalsh(H)
        min_eigenvalue = eigenvalues.min()
        max_eigenvalue = eigenvalues.max()
        eigen_ratio = min_eigenvalue / max_eigenvalue
        
        # Confidence based on eigenvalue ratio and residual
        residual = np.sum((r @ source.T + t[:, np.newaxis] - target) ** 2) / len(source)
        confidence[ii] = max(0.0, 1.0 - 10 * eigen_ratio - 5 * residual)
        
        transforms[ii] = get_transform_matrix(r, t)
    
    np.save("transforms.npy", transforms)
    np.save("confidence.npy", confidence)

if __name__ == "__main__":
    main()
```