import numpy as np
import os
import json
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

def load_point_clouds(pair_idx):
    source_path = f"data/pairs/{pair_idx:02d}/source.npy"
    target_path = f"data/pairs/{pair_idx:02d}/target.npy"
    source = np.load(source_path)
    target = np.load(target_path)
    return source, target

def compute_pca_alignment(source, target):
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    cov_matrix = np.dot(source_centered.T, target_centered)
    U, S, Vh = np.linalg.svd(cov_matrix)
    R = np.dot(Vh.T, U.T)
    if np.linalg.det(R) < 0:
        R = np.dot(R, np.diag([-1, -1, 1]))
    t = target_centroid - np.dot(R, source_centroid)
    return R, t

def icp(source, target, max_iter=100, tolerance=1e-6):
    source_tree = cKDTree(source)
    R, t = np.eye(3), np.zeros(3)
    for _ in range(max_iter):
        dist, idx = source_tree.query(target)
        correspondences = np.column_stack((source, target))
        correspondences = correspondences[dist < tolerance]
        if len(correspondences) < 3:
            break
        source_corr = correspondences[:, :3]
        target_corr = correspondences[:, 3:]
        centroid_source = np.mean(source_corr, axis=0)
        centroid_target = np.mean(target_corr, axis=0)
        source_centered = source_corr - centroid_source
        target_centered = target_corr - centroid_target
        cov = np.dot(source_centered.T, target_centered)
        U, S, Vh = np.linalg.svd(cov)
        R = np.dot(Vh.T, U.T)
        if np.linalg.det(R) < 0:
            R = np.dot(R, np.diag([-1, -1, 1]))
        t = centroid_target - np.dot(R, centroid_source)
        source_tree = cKDTree(np.dot(R, source.T).T + t)
    return R, t

def compute_hessian(source, target, R, t):
    source_aligned = np.dot(R, source.T).T + t
    diff = source_aligned - target
    H = np.zeros((6, 6))
    for i in range(len(diff)):
        d = diff[i]
        H[0:3, 0:3] += np.outer(d, d)
        H[0:3, 3:6] += np.outer(d, source[i])
        H[3:6, 0:3] += np.outer(d, source[i])
        H[3:6, 3:6] += np.outer(source[i], source[i])
    return H

def compute_confidence(source, target, R, t, H):
    eigenvalues = np.linalg.eigvalsh(H)
    if eigenvalues[-1] < 1e-6:
        return 0.2
    residual = np.mean((np.dot(R, source.T).T + t - target) ** 2)
    if residual > 1e-4:
        return 0.3
    return 0.9

def main():
    with open("data/task_info.json") as f:
        task_info = json.load(f)
    num_pairs = task_info["num_pairs"]
    transforms = np.zeros((num_pairs, 4, 4))
    confidence = np.zeros(num_pairs)
    
    for i in range(num_pairs):
        source, target = load_point_clouds(i)
        R, t = compute_pca_alignment(source, target)
        R, t = icp(source, target, max_iter=100, tolerance=1e-5)
        H = compute_hessian(source, target, R, t)
        conf = compute_confidence(source, target, R, t, H)
        transforms[i, :3, :3] = R
        transforms[i, :3, 3] = t
        transforms[i, 3, 3] = 1.0
        confidence[i] = conf
    
    np.save("transforms.npy", transforms)
    np.save("confidence.npy", confidence)

if __name__ == "__main__":
    main()