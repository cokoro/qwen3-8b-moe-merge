import os
import numpy as np
import json
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.linalg import orthogonal_procrustes

def load_pair(ii):
    source = np.load(f"data/pairs/{ii:02d}/source.npy")
    target = np.load(f"data/pairs/{ii:02d}/target.npy")
    return source, target

def align_principal_axes(source, target):
    # PCA on source and target
    cov_source = np.cov(source, rowvar=False)
    eig_source = np.linalg.eigh(cov_source)
    rot_source = eig_source[1][:, eig_source[0].argsort()[::-1]]
    cov_target = np.cov(target, rowvar=False)
    eig_target = np.linalg.eigh(cov_target)
    rot_target = eig_target[1][:, eig_target[0].argsort()[::-1]]
    # Align source to target
    R = rot_target.T @ rot_source
    return R

def rigid_transform_from_points(A, B):
    # SVD of the cross-covariance matrix
    A_centroid = A.mean(axis=0)
    B_centroid = B.mean(axis=0)
    A_centered = A - A_centroid
    B_centered = B - B_centroid
    H = A_centered.T @ B_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # Ensure right-hand coordinate system
    if np.linalg.det(R) < 0:
        R[2, :] *= -1
    t = B_centroid - R @ A_centroid
    return R, t

def icp(source, target, max_iter=20, eps=1e-6):
    # Initialize with PCA alignment
    R, t = rigid_transform_from_points(source, target)
    for _ in range(max_iter):
        # Find nearest neighbors
        tree = cKDTree(target)
        dist, idx = tree.query(source, k=1)
        # Compute transformation
        A = source
        B = target[idx]
        R, t = rigid_transform_from_points(A, B)
        # Check convergence
        if np.max(dist) < eps:
            break
    return R, t

def get_registration_info(R, t, source, target):
    # Compute Jacobian of residual
    A = source
    B = target
    n = A.shape[0]
    J = np.zeros((3 * n, 6))
    for i in range(n):
        x, y, z = A[i]
        J[3*i:3*i+3, 0:3] = np.array([-y, x, 0]).reshape(-1, 1)
        J[3*i:3*i+3, 3:6] = np.array([-t[1], -t[2], -t[0]]).reshape(-1, 1)
    # Compute Hessian
    H = J.T @ J
    return H

def get_confidence(R, t, source, target):
    H = get_registration_info(R, t, source, target)
    eigenvalues = np.linalg.eigvalsh(H)
    # Check if the smallest eigenvalue is tiny
    degenerate = eigenvalues[-1] / eigenvalues[0] > 1e-3
    # Check for repeated structure
    if degenerate:
        return 0.1
    return 0.9

def main():
    with open("data/task_info.json", "r") as f:
        task_info = json.load(f)
    num_pairs = task_info["num_pairs"]
    transforms = np.zeros((num_pairs, 4, 4))
    confidence = np.zeros(num_pairs)
    for ii in range(num_pairs):
        source, target = load_pair(ii)
        R, t = icp(source, target)
        transforms[ii, :3, :3] = R
        transforms[ii, :3, 3] = t
        transforms[ii, 3, 3] = 1.0
        conf = get_confidence(R, t, source, target)
        confidence[ii] = conf
    np.save("transforms.npy", transforms)
    np.save("confidence.npy", confidence)

if __name__ == "__main__":
    main()