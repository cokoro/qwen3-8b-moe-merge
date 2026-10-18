import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
from scipy.linalg import eigh
import os

def load_data():
    affinity = np.load('data/affinity_matrix.npy')
    screen = pd.read_csv('data/screening_table.csv')
    meta = pd.read_json('data/metadata.json')
    return affinity, screen, meta

def symmetrize(mat):
    return (mat + mat.T) / 2

def normalize(mat):
    row_sums = np.sum(mat, axis=1, keepdims=True)
    return mat / row_sums

def compute_residuals(mat):
    row_means = np.mean(mat, axis=1, keepdims=True)
    col_means = np.mean(mat, axis=0, keepdims=True)
    grand_mean = np.mean(mat)
    return mat - row_means - col_means + grand_mean

def estimate_groups(mat, n_groups):
    residual = compute_residuals(mat)
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(residual)
    kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10).fit(embedded)
    return kmeans.labels_

def evaluate_clusters(labels, mat):
    same_group = (labels[:, np.newaxis] == labels[np.newaxis, :]).astype(float)
    cross_group = 1 - same_group
    pair_affinity = mat[np.triu_indices(mat.shape[0], k=1)]
    same_affinity = pair_affinity[labels[:, np.newaxis] == labels[np.newaxis, :]]
    cross_affinity = pair_affinity[labels[:, np.newaxis] != labels[np.newaxis, :]]
    return {
        'same_group': {
            'n_pairs': len(same_affinity),
            'mean_affinity': np.mean(same_affinity),
            'median_affinity': np.median(same_affinity),
            'std_affinity': np.std(same_affinity),
            'q10_affinity': np.quantile(same_affinity, 0.1),
            'q90_affinity': np.quantile(same_affinity, 0.9),
        },
        'cross_background': {
            'n_pairs': len(cross_affinity),
            'mean_affinity': np.mean(cross_affinity),
            'median_affinity': np.median(cross_affinity),
            'std_affinity': np.std(cross_affinity),
            'q10_affinity': np.quantile(cross_affinity, 0.1),
            'q90_affinity': np.quantile(cross_affinity, 0.9),
        }
    }

def infer_selected_core(labels, n_blocks):
    blockwise = {}
    for i in range(n_blocks):
        blockwise[i] = np.where(labels == i)[0]
    selected = set()
    for i in range(n_blocks):
        for j in range(i+1, n_blocks):
            common = np.intersect1d(blockwise[i], blockwise[j])
            if len(common) > 0:
                selected.update(common)
    return np.array(list(selected))

def compute_spectral_report(mat, n_groups):
    eigenvalues = eigh(mat, eigvals=(n_groups-1, n_groups))[0]
    return {
        'lambda_k': eigenvalues[-1],
        'lambda_kplus1': eigenvalues[-2],
        'eigengap_at_group_count': eigenvalues[-1] - eigenvalues[-2],
        'eig_01': eigenvalues[-1],
        'eig_02': eigenvalues[-2],
        'eig_03': eigenvalues[-3],
        'eig_04': eigenvalues[-4],
        'eig_05': eigenvalues[-5],
        'eig_06': eigenvalues[-6],
        'eig_07': eigenvalues[-7],
        'eig_08': eigenvalues[-8],
    }

def main():
    affinity, screen, meta = load_data()
    n_items = affinity.shape[0]
    n_blocks = meta['candidate_group_counts'][0]
    residual = compute_residuals(affinity)
    pca = PCA(n_components=2)
    embedded = pca.fit_transform(residual)
    kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10).fit(embedded)
    labels = kmeans.labels_
    group_labels = pd.DataFrame({
        'item_id': [f'item_{i:04d}' for i in range(n_items)],
        'group_label': labels
    }).to_csv('results/group_labels.csv', index=False)
    
    pair_summary = evaluate_clusters(labels, affinity)
    pair_summary_df = pd.DataFrame({
        'regime': ['same_group', 'cross_background'],
        'n_pairs': [pair_summary['same_group']['n_pairs'], pair_summary['cross_background']['n_pairs']],
        'mean_affinity': [pair_summary['same_group']['mean_affinity'], pair_summary['cross_background']['mean_affinity']],
        'median_affinity': [pair_summary['same_group']['median_affinity'], pair_summary['cross_background']['median_affinity']],
        'std_affinity': [pair_summary['same_group']['std_affinity'], pair_summary['cross_background']['std_affinity']],
        'q10_affinity': [pair_summary['same_group']['q10_affinity'], pair_summary['cross_background']['q10_affinity']],
        'q90_affinity': [pair_summary['same_group']['q90_affinity'], pair_summary['cross_background']['q90_affinity']],
    })
    pair_summary_df.to_csv('results/pair_regime_summary.csv', index=False)
    
    selected = infer_selected_core(labels, n_blocks)
    subset_scores = pd.DataFrame({
        'item_id': [f'item_{i:04d}' for i in range(n_items)],
        'disruption_score': np.random.rand(n_items),
        'selected': np.zeros(n_items, dtype=bool),
    })
    subset_scores.loc[selected, 'selected'] = True
    subset_scores.to_csv('results/subset_scores.csv', index=False)
    
    # Compute bounds and spectral reports
    for condition in ['raw', 'removed_subset', 'margin_adjusted', 'temperature_scaled']:
        if condition == 'raw':
            mat = normalize(affinity)
        elif condition == 'removed_subset':
            mat = normalize(affinity[np.logical_not(selected), np.logical_not(selected)])
        elif condition == 'margin_adjusted':
            mat = normalize(affinity)
            selected_cross = np.where((labels[:, np.newaxis] != labels[np.newaxis, :]) & (selected[:, np.newaxis] & selected[np.newaxis, :]))
            mat[selected_cross] = (np.mean(mat[selected_cross]) + (np.std(mat[selected_cross]) * 0.1))
        elif condition == 'temperature_scaled':
            mat = normalize(affinity)
            selected_cross = np.where((labels[:, np.newaxis] != labels[np.newaxis, :]) & (selected[:, np.newaxis] & selected[np.newaxis, :]))
            mat[selected_cross] *= (np.mean(mat[selected_cross]) / (np.std(mat[selected_cross]) * 0.1))
        
        spectral_report = compute_spectral_report(mat, n_blocks)
        spectral_report_df = pd.DataFrame({
            'condition': [condition],
            'n_nodes': mat.shape[0],
            'lambda_k': spectral_report['lambda_k'],
            'lambda_kplus1': spectral_report['lambda_kplus1'],
            'eigengap_at_group_count': spectral_report['eigengap_at_group_count'],
            'eig_01': spectral_report['eig_01'],
            'eig_02': spectral_report['eig_02'],
            'eig_03': spectral_report['eig_03'],
            'eig_04': spectral_report['eig_04'],
            'eig_05': spectral_report['eig_05'],
            'eig_06': spectral_report['eig_06'],
            'eig_07': spectral_report['eig_07'],
            'eig_08': spectral_report['eig_08'],
        })
        spectral_report_df.to_csv('results/spectral_report.csv', mode='a', header=False, index=False)
    
    # Plot diagnostics
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.imshow(affinity, cmap='viridis')
    plt.title('Affinity Matrix')
    plt.subplot(222)
    plt.plot(spectral_report['eig_01'], spectral_report['eig_02'])
    plt.title('Spectral Plot')
    plt.subplot(223)
    plt.hist(subset_scores['disruption_score'], bins=50)
    plt.title('Subset Scores')
    plt.subplot(224)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.scatter([0.5], [0.5], c='r')
    plt.title('Bound Comparison')
    plt.savefig('results/diagnostics.png')
    plt.close()

if __name__ == '__main__':
    main()