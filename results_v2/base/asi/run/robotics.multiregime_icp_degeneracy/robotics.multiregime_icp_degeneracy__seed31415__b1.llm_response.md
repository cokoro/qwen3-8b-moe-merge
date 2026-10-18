```python
import numpy as np
import os
import json
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

def load_data():
    with open('data/task_info.json') as f:
        task_info = json.load(f)
    num_pairs = task_info['num_pairs']
    transforms = np.zeros((num_pairs, 4, 4))
    confidence = np.zeros(num_pairs)
    for ii in range(num_pairs):
        source = np.load(f'data/pairs/{ii:02d}/source.npy')
        target = np.load(f'data/pairs/{ii:02d}/target.npy')
        # Coarse alignment using PCA
        pca_source = PCA(n_components=3).fit(source)
        pca_target = PCA(n_components=3).fit(target)
        R_coarse, _ = orthogonal_procrustes(pca_source.components_, pca_target.components_)
        # Refine with ICP
        def icp_cost(R, t, source, target):
            transformed = R @ source.T + t
            tree = cKDTree(target)
            dist, idx = tree.query(transformed.T)
            cost = np.sum(dist)
            return cost
        def icp_update(R, t, source, target):
            transformed = R @ source.T + t
            tree = cKDTree(target)
            dist, idx = tree.query(transformed.T)
            correspondences = np.column_stack((source, target[idx]))
            C = np.dot(correspondences[:, :3], correspondences[:, 3:6].T)
            U, S, Vt = np.linalg.svd(C)
            R_update = np.dot(U, Vt)
            t_update = np.mean(correspondences[:, 3:6] - np.dot(R_update, correspondences[:, :3].T), axis=0)
            return R_update, t_update
        R, t = R_coarse, np.mean(target, axis=0) - np.mean(source, axis=0)
        for _ in range(10):
            R, t = icp_update(R, t, source, target)
        # Compute Hessian for geometric degeneracy
        transformed = R @ source.T + t
        tree = cKDTree(target)
        dist, idx = tree.query(transformed.T)
        W = np.zeros((source.shape[0], target.shape[0]))
        W[np.arange(source.shape[0]), idx] = 1
        A = np.dot(W, target - transformed)
        A = np.vstack((A, np.zeros((3, source.shape[0]))))
        A = np.vstack((A, np.zeros((source.shape[0], 3))))
        A = np.vstack((A, np.zeros((3, 3))))
        H = np.dot(A.T, A)
        eigenvalues = np.linalg.eigvalsh(H)
        degeneracy = eigenvalues[-1] / eigenvalues[0]
        # Check for ambiguity
        initializations = []
        for _ in range(5):
            R_rand = np.random.rand(3, 3)
            R_rand = np.dot(R_rand, R_rand.T)
            t_rand = np.random.rand(3)
            transformed_rand = R_rand @ source.T + t_rand
            tree_rand = cKDTree(target)
            dist_rand, idx_rand = tree_rand.query(transformed_rand.T)
            W_rand = np.zeros((source.shape[0], target.shape[0]))
            W_rand[np.arange(source.shape[0]), idx_rand] = 1
            A_rand = np.dot(W_rand, target - transformed_rand)
            A_rand = np.vstack((A_rand, np.zeros((3, source.shape[0]))))
            A_rand = np.vstack((A_rand, np.zeros((source.shape[0], 3))))
            A_rand = np.vstack((A_rand, np.zeros((3, 3))))
            H_rand = np.dot(A_rand.T, A_rand)
            eigenvalues_rand = np.linalg.eigvalsh(H_rand)
            degeneracy_rand = eigenvalues_rand[-1] / eigenvalues_rand[0]
            if degeneracy_rand < 1e-3:
                continue
            R_rand, t_rand = icp_update(R_rand, t_rand, source, target)
            transformed_rand = R_rand @ source.T + t_rand
            tree_rand = cKDTree(target)
            dist_rand, idx_rand = tree_rand.query(transformed_rand.T)
            cost_rand = np.sum(dist_rand)
            initializations.append((R_rand, t_rand, cost_rand))
        if len(initializations) < 2:
            ambiguity = 0
        else:
            costs = [c for (R, t, c) in initializations]
            min_cost = np.min(costs)
            std_cost = np.std(costs)
            ambiguity = np.sum(np.abs(costs - min_cost) < 2 * std_cost) / len(costs)
        # Confidence calculation
        confidence_val = 1.0
        if degeneracy > 1e-2:
            confidence_val *= 0.5
        if ambiguity > 0.5:
            confidence_val *= 0.5
        confidence[ii] = confidence_val
        # Store transform
        transforms[ii, :3, :3] = R
        transforms[ii, :3, 3] = t
        transforms[ii, 3, 3] = 1.0
    np.save('transforms.npy', transforms)
    np.save('confidence.npy', confidence)

load_data()
```