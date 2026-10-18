import numpy as np
import os
import json

def load_data():
    with open('data/task_info.json') as f:
        meta = json.load(f)
    num_pairs = meta['num_pairs']
    return num_pairs, meta['units'], meta['convention']

def pca_alignment(src, tgt):
    # Compute PCA for each point cloud to estimate initial rotation
    src_mean = np.mean(src, axis=0)
    tgt_mean = np.mean(tgt, axis=0)
    src_centered = src - src_mean
    tgt_centered = tgt - tgt_mean
    # Compute covariance matrix between src and tgt
    cov = np.dot(src_centered.T, tgt_centered) / src_centered.shape[0]
    # SVD to get rotation matrix
    U, S, Vt = np.linalg.svd(cov)
    R = np.dot(Vt.T, U.T)
    # Ensure right-hand coordinate system
    if np.linalg.det(R) < 0:
        R = np.dot(R, np.diag([-1, -1, 1]))
    t = tgt_mean - np.dot(R, src_mean)
    return R, t

def icp(src, tgt, init_R, init_t, max_iter=50, tol=1e-4):
    R, t = init_R, init_t
    for _ in range(max_iter):
        # Find nearest neighbors
        src_pts = src
        tgt_pts = tgt
        # Compute correspondences (simplified)
        dists = np.linalg.norm(src_pts[:, np.newaxis, :] - tgt_pts[np.newaxis, :, :], axis=2)
        idx = np.argmin(dists, axis=1)
        # Compute optimal R, t
        src_correspond = src_pts
        tgt_correspond = tgt_pts[idx]
        # Compute cross-covariance matrix
        H = np.dot(src_correspond.T, src_correspond)
        S = np.dot(src_correspond.T, tgt_correspond)
        # SVD to solve for R and t
        U, S, Vt = np.linalg.svd(S)
        R = np.dot(U, Vt)
        t = np.mean(tgt_correspond - np.dot(R, src_correspond), axis=0)
        # Check convergence
        if np.linalg.norm(np.dot(R, src_correspond) + t - tgt_correspond) < tol:
            break
    return R, t

def compute_confidence(R, t, src, tgt, num_trials=5):
    # Compute residual
    residual = np.linalg.norm(np.dot(R, src) + t - tgt, axis=1)
    residual_mean = np.mean(residual)
    residual_std = np.std(residual)
    # Check if the Hessian is singular (eigenvalues)
    H = np.dot(src.T, src) - np.dot(src.T, np.dot(R, tgt) + t)
    eigvals = np.linalg.eigvalsh(H)
    min_eig = np.min(eigvals)
    max_eig = np.max(eigvals)
    # Check if eigenvalues are small (degenerate)
    degenerate = min_eig < 1e-4 * max_eig
    # Check if multiple solutions are possible (repeated structure)
    # (this is simplified and would require more sophisticated analysis)
    ambiguous = False
    # Combine to compute confidence
    confidence = 0.5 + 0.5 * (1 - degenerate) * (1 - ambiguous)
    return confidence

def main():
    num_pairs, units, convention = load_data()
    transforms = np.zeros((num_pairs, 4, 4), dtype=np.float64)
    confidence = np.zeros(num_pairs, dtype=np.float64)
    for ii in range(num_pairs):
        src = np.load(f'data/pairs/{ii:02d}/source.npy')
        tgt = np.load(f'data/pairs/{ii:02d}/target.npy')
        # Step 1: Coarse alignment using PCA
        R_coarse, t_coarse = pca_alignment(src, tgt)
        # Step 2: Refine with ICP
        R, t = icp(src, tgt, R_coarse, t_coarse)
        # Step 3: Compute confidence
        conf = compute_confidence(R, t, src, tgt)
        # Store results
        transforms[ii, :3, :3] = R
        transforms[ii, :3, 3] = t
        transforms[ii, 3, 3] = 1.0
        confidence[ii] = conf
    np.save('transforms.npy', transforms)
    np.save('confidence.npy', confidence)

if __name__ == '__main__':
    main()