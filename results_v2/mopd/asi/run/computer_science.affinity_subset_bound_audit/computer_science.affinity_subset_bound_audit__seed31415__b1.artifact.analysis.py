import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.stats import zscore
import json
import os

# Load data
affinity_matrix = np.load('data/affinity_matrix.npy')
screening_table = pd.read_csv('data/screening_table.csv')
metadata = json.load(open('data/metadata.json'))

# Step 1: Load and symmetrize affinity matrix
A = (affinity_matrix + affinity_matrix.T) / 2
A = csr_matrix(A)
n = A.shape[0]

# Step 2: Remove nuisance and normalize
A_diag_zero = A.copy()
A_diag_zero.setdiag(0)
degrees = np.array(A_diag_zero.sum(axis=1)).flatten()
A_diag_zero = A_diag_zero - csr_matrix(np.outer(degrees, degrees) / n)
A_diag_zero = A_diag_zero / np.sqrt(n)
A_diag_zero = A_diag_zero.toarray()
A_diag_zero = normalize(A_diag_zero, norm='l1', axis=1)

# Step 3: Cluster
k = metadata['candidate_group_counts'][0]
eigenvalues, eigenvectors = np.linalg.eig(A_diag_zero)
eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(eigenvectors[:, :k])
group_labels = kmeans.labels_

# Step 4: Identify selected subset
# Compute cross-block residual scores
blockwise_medians = np.array([np.median(A_diag_zero[group_labels == i, group_labels != i], axis=0) for i in range(k)])
selected = np.zeros(n, dtype=bool)
for i in range(n):
    block = group_labels[i]
    residual = A_diag_zero[i, group_labels != block] - blockwise_medians[block][group_labels != block]
    selected[i] = np.sum(np.abs(residual) > np.median(np.abs(residual))) > k - 1

# Step 5: Estimate regimes
same_group = np.zeros(n, dtype=bool)
selected_cross = np.zeros(n, dtype=bool)
cross_background = np.zeros(n, dtype=bool)
for i in range(n):
    for j in range(n):
        if i < j:
            if group_labels[i] == group_labels[j]:
                same_group[i] = True
                same_group[j] = True
            elif selected[i] and selected[j]:
                selected_cross[i] = True
                selected_cross[j] = True
            else:
                cross_background[i] = True
                cross_background[j] = True

# Step 6: Build adjusted matrices
raw_normalized = normalize(A, norm='l1', axis=1)
removed_normalized = raw_normalized[~selected, ~selected]
selected_indices = np.where(selected)[0]
kept_indices = np.where(~selected)[0]
margin_adjusted = raw_normalized.copy()
margin_adjusted[selected_indices, selected_indices] = (np.mean(raw_normalized[selected_indices, selected_indices]) + np.mean(raw_normalized[~selected_indices, ~selected_indices])) / 2
temperature_scaled = raw_normalized.copy()
temperature_scaled[selected_indices, selected_indices] *= (np.mean(raw_normalized[selected_indices, selected_indices]) / np.mean(raw_normalized[~selected_indices, ~selected_indices]))

# Step 7: Compute spectral report
eigenvalues_raw = np.linalg.eigvals(raw_normalized)
eigenvalues_removed = np.linalg.eigvals(removed_normalized)
eigenvalues_margin = np.linalg.eigvals(margin_adjusted)
eigenvalues_temp = np.linalg.eigvals(temperature_scaled)

# Step 8: Compute pair regime summary
pair_regime_summary = pd.DataFrame(index=['same_group', 'cross_background', 'selected_cross'])
pair_regime_summary['n_pairs'] = [np.sum(same_group), np.sum(cross_background), np.sum(selected_cross)]
pair_regime_summary['mean_affinity'] = [np.mean(A[same_group, same_group]), np.mean(A[cross_background, cross_background]), np.mean(A[selected_cross, selected_cross])]
pair_regime_summary['median_affinity'] = [np.median(A[same_group, same_group]), np.median(A[cross_background, cross_background]), np.median(A[selected_cross, selected_cross])]
pair_regime_summary['std_affinity'] = [np.std(A[same_group, same_group]), np.std(A[cross_background, cross_background]), np.std(A[selected_cross, selected_cross])]
pair_regime_summary['q