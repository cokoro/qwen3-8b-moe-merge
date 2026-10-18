```bash
#!/bin/bash

# Generate an analysis script and run it.
# This script runs a coarse alignment (PCA), refines with ICP,
# and evaluates the reliability of each estimate.
# It outputs transforms.npy and confidence.npy.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def load_pair(ii):
    source = np.load(f"data/pairs/{ii:02d}/source.npy")
    target = np.load(f"data/pairs/{ii:02d}/target.npy")
    return source, target

def align_pca(source, target):
    # Compute centroids.
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)

    # Remove centroids.
    source = source - source_centroid
    target = target - target_centroid

    # Compute covariance.
    covariance = source.T @ target

    # Compute SVD.
    u, s, vh = np.linalg.svd(covariance, full_matrices=True)

    # Compute rotation.
    rotation = u @ vh

    # Compute translation.
    translation = target_centroid - rotation @ source_centroid

    return rotation, translation

def icp(source, target, max_iter=50, tolerance=1e-6):
    source = source.copy()
    target = target.copy()

    for _ in range(max_iter):
        # Compute nearest neighbours.
        _, indices = cdist(source, target).argmin(axis=1)

        # Compute cross-covariance.
        cross_cov = np.cov(source.T, target.T)

        # Compute SVD.
        u, s, vh = np.linalg.svd(cross_cov, full_matrices=True)

        # Compute rotation.
        rotation = u @ vh

        # Compute translation.
        translation = target.mean(axis=0) - rotation @ source.mean(axis=0)

        # Apply transformation.
        source = rotation @ source.T + translation
        source = source.T

        # Check convergence.
        if np.linalg.norm(translation) < tolerance:
            break

    return rotation, translation

def evaluate_confidence(source, target, rotation, translation, num_restarts=5):
    # Compute the residual.
    residual = np.linalg.norm(source @ rotation.T + translation - target)

    # Compute the Hessian.
    hessian = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            hessian[i, j] = np.cov(source.T, target.T)[i // 3, j // 3]

    # Compute the smallest eigenvalue.
    eigenvalues = np.linalg.eigvalsh(hessian)
    smallest_eigenvalue = eigenvalues.min()

    # Compute the largest eigenvalue.
    largest_eigenvalue = eigenvalues.max()

    # Compute the condition number.
    condition_number = largest_eigenvalue / smallest_eigenvalue

    # Compute the number of distinct initialisations.
    for _ in range(num_restarts):
        rotation_restart, translation_restart = icp(source, target)
        residual_restart = np.linalg.norm(source @ rotation_restart.T + translation_restart - target)
        if abs(residual_restart - residual) > 1e-6:
            return 0.0

    return 1.0

def main():
    with open("data/task_info.json") as f:
        task_info = json.load(f)

    num_pairs = task_info["num_pairs"]
    units = task_info["units"]
    transform_convention = task_info["transform_convention"]
    output_transforms = np.zeros((num_pairs, 4, 4), dtype=np.float64)
    output_confidence = np.zeros(num_pairs, dtype=np.float64)

    for ii in range(num_pairs):
        source, target = load_pair(ii)
        rotation, translation = align_pca(source, target)
        rotation, translation = icp(source, target)
        confidence = evaluate_confidence(source, target, rotation, translation)
        output_transforms[ii, :3, :3] = rotation
        output_transforms[ii, :3, 3] = translation
        output_transforms[ii, 3, 3] = 1.0
        output_confidence[ii] = confidence

    np.save("transforms.npy", output_transforms)
    np.save("confidence.npy", output_confidence)

if __name__ == "__main__":
    main()
```