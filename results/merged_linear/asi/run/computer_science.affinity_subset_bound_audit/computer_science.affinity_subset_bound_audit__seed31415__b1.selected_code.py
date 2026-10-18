import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Load the affinity matrix and symmetrize it
affinity_matrix = np.load('data/affinity_matrix.npy')
affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2

# Load the screening table and metadata
screening_table = pd.read_csv('data/screening_table.csv')
metadata = pd.read_json('data/metadata.json')

# Compute degrees and form a residual matrix
degrees = np.sum(affinity_matrix, axis=1)
residual_matrix = affinity_matrix - np.outer(degrees, degrees) / np.sum(degrees)

# Remove nuisance by regressing out the leading non-community popularity vector
pca = PCA(n_components=1)
pca.fit(residual_matrix)
residual_matrix -= pca.transform(residual_matrix)

# Cluster rows of the residual normalized affinity matrix
k = 8  # Use the published group count
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
group_labels = kmeans.fit_predict(residual_matrix)

# Compute cross-block residual scores and identify a broad-support dense core
blockwise_residuals = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        if i != j:
            blockwise_residuals[i, j] = np.mean(residual_matrix[group_labels == i, group_labels == j])

selected_indices = []
for i in range(k):
    for j in range(k):
        if i != j and blockwise_residuals[i, j] > 0.5:
            selected_indices.extend(group_labels[group_labels == i])

selected_indices = np.unique(selected_indices)

# Estimate regimes from recovered labels and selected flags
same_group_pairs = []
cross_background_pairs = []
selected_cross_pairs = []
for i in range(affinity_matrix.shape[0]):
    for j in range(i + 1, affinity_matrix.shape[0]):
        if group_labels[i] == group_labels[j]:
            same_group_pairs.append((i, j))
        elif i not in selected_indices or j not in selected_indices:
            cross_background_pairs.append((i, j))
        else:
            selected_cross_pairs.append((i, j))

# Build adjusted matrices
raw_normalized = normalize(affinity_matrix)
removed_normalized = normalize(affinity_matrix[selected_indices, selected_indices])
margin_adjusted_normalized = affinity_matrix.copy()
for i, j in selected_cross_pairs:
    margin_adjusted_normalized[i, j] = margin_adjusted_normalized[i, j] - (np.median(affinity_matrix[group_labels == group_labels[i], group_labels == group_labels[j]]) - np.median(affinity_matrix[group_labels == group_labels[i], group_labels == group_labels[j]]))
margin_adjusted_normalized[selected_indices, selected_indices] = normalize(affinity_matrix[selected_indices, selected_indices])
temperature_scaled_normalized = affinity_matrix.copy()
for i, j in selected_cross_pairs:
    temperature_scaled_normalized[i, j] = temperature_scaled_normalized[i, j] * (np.sum(affinity_matrix[group_labels == group_labels[i], group_labels == group_labels[j]]) / np.sum(affinity_matrix[group_labels == group_labels[i], group_labels == group_labels[j]]))
temperature_scaled_normalized[selected_indices, selected_indices] = normalize(affinity_matrix[selected_indices, selected_indices])

# Compute top normalized-affinity eigenvalues and report lambda_k, lambda_kplus1, and the eigengap at the chosen group count
eigenvalues = np.linalg.eigvals(temperature_scaled_normalized)
lambda_k = eigenvalues[-k]
lambda_kplus1 = eigenvalues[-k - 1]
eigengap_at_group_count = lambda_k - lambda_kplus1

# Compute spectral report
spectral_report = pd.DataFrame({
    'condition': ['raw', 'removed_subset', 'margin_adjusted', 'temperature_scaled'],
    'n_nodes': [affinity_matrix.shape[0], affinity_matrix.shape[0], affinity_matrix.shape[0], affinity_matrix.shape[0]],
    'lambda_k': [eigenvalues[-k], eigenvalues[-k], eigenvalues[-k], eigenvalues[-k]],
    'lambda_kplus1': [eigenvalues[-k - 1], eigenvalues[-k - 1], eigenvalues[-k - 1], eigenvalues[-k - 1]],
    'eigengap_at_group_count': [eigengap_at_group_count, eigengap_at_group_count, eigengap_at_group_count, eigengap_at_group_count],
    'eig_01': [eigenvalues[-1], eigenvalues[-1], eigenvalues[-1], eigenvalues[-1]],
    'eig_02': [eigenvalues[-2], eigenvalues[-2], eigenvalues[-2], eigenvalues[-2]],
    'eig_03': [eigenvalues[-3], eigenvalues[-3], eigenvalues[-3], eigenvalues[-3]],
    'eig_04': [eigenvalues[-4], eigenvalues[-4], eigenvalues[-4], eigenvalues[-4]],
    'eig_05': [eigenvalues[-5], eigenvalues[-5], eigenvalues[-5], eigenvalues[-5]],
    'eig_06': [eigenvalues[-6], eigenvalues[-6], eigenvalues[-6], eigenvalues[-6]],
    'eig_07': [eigenvalues[-7], eigenvalues[-7], eigenvalues[-7], eigenvalues[-7]],
    'eig_08': [eigenvalues[-8], eigenvalues[-8], eigenvalues[-8], eigenvalues[-8]]
})

# Compute pair regime summary
same_group_mean = np.mean(affinity_matrix[np.array(same_group_pairs)])
same_group_median = np.median(affinity_matrix[np.array(same_group_pairs)])
same_group_std = np.std(affinity_matrix[np.array(same_group_pairs)])
same_group_q10 = np.percentile(affinity_matrix[np.array(same_group_pairs)], 10)
same_group_q90 = np.percentile(affinity_matrix[np.array(same_group_pairs)], 90)

cross_background_mean = np.mean(affinity_matrix[np.array(cross_background_pairs)])
cross_background_median = np.median(affinity_matrix[np.array(cross_background_pairs)])
cross_background_std = np.std(affinity_matrix[np.array(cross_background_pairs)])
cross_background_q10 = np.percentile(affinity_matrix[np.array(cross_background_pairs)], 10)
cross_background_q90 = np.percentile(affinity_matrix[np.array(cross_background_pairs)], 90)

selected_cross_mean = np.mean(affinity_matrix[np.array(selected_cross_pairs)])
selected_cross_median = np.median(affinity_matrix[np.array(selected_cross_pairs)])
selected_cross_std = np.std(affinity_matrix[np.array(selected_cross_pairs)])
selected_cross_q10 = np.percentile(affinity_matrix[np.array(selected_cross_pairs)], 10)
selected_cross_q90 = np.percentile(affinity_matrix[np.array(selected_cross_pairs)], 90)

pair_regime_summary = pd.DataFrame({
    'regime': ['same_group', 'cross_background', 'selected_cross'],
    'n_pairs': [len(same_group_pairs), len(cross_background_pairs), len(selected_cross_pairs)],
    'mean_affinity': [same_group_mean, cross_background_mean, selected_cross_mean],
    'median_affinity': [same_group_median, cross_background_median, selected_cross_median],
    'std_affinity': [same_group_std, cross_background_std, selected_cross_std],
    'q10_affinity': [same_group_q10, cross_background_q10, selected_cross_q10],
    'q90_affinity': [same_group_q90, cross_background_q90, selected_cross_q90]
})

# Compute bound comparison
delta = 0.25 * (pair_regime_summary['std_affinity'].mean())
n_effective_per_group = len(selected_indices) / k
selected_per_group_effective = len(selected_indices) / k
alpha_hat = selected_cross_mean
beta_hat = cross_background_mean
gamma_hat = selected_cross_mean
lambda_formula = (1 - alpha_hat) / ((1 - alpha_hat) + n_effective_per_group * alpha_hat + n_effective_per_group * (k - 1) * beta_hat)
lambda_k = eigenvalues[-k]
lambda_kplus1 = eigenvalues[-k - 1]
eigengap_at_group_count = lambda_k - lambda_kplus1
error_bound = 4 * delta / (1 - lambda_formula) + 8 * delta
improvement_vs_raw = 1 - error_bound / lambda_k
margin_value = ((1 - alpha_hat) + n_effective_per_group * alpha_hat + (n_effective_per_group - selected_per_group_effective) * (k - 1) * beta_hat) * (gamma_hat - beta_hat) / (((1 - alpha_hat) + n_effective_per_group * alpha_hat + n_effective_per_group * (k - 1) * beta_hat) ** 2 * ((1 - alpha_hat) + n_effective_per_group * alpha_hat + n_effective_per_group * (k - 1) * beta_hat))
temperature_factor = ((1 - alpha_hat) + n_effective_per_group * alpha_hat + n_effective_per_group * (k - 1) * beta_hat) / ((1 - alpha_hat) + n_effective_per_group * alpha_hat + n_effective_per_group * (k - 1) * beta_hat) * (beta_hat / gamma_hat)

bound_comparison = pd.DataFrame({
    'condition': ['raw', 'removed_subset', 'margin_adjusted', 'temperature_scaled'],
    'n_effective_per_group': [n_effective_per_group, n_effective_per_group, n_effective_per_group, n_effective_per_group],
    'selected_per_group_effective': [selected_per_group_effective, selected_per_group_effective, selected_per_group_effective, selected_per_group_effective],
    'alpha_hat': [alpha_hat, alpha_hat, alpha_hat, alpha_hat],
    'beta_hat': [beta_hat, beta_hat, beta_hat, beta_hat],
    'gamma_hat': [gamma_hat, gamma_hat, gamma_hat, gamma_hat],
    'lambda_formula': [lambda_formula, lambda_formula, lambda_formula, lambda_formula],
    'spectral_lambda_kplus1': [lambda_k, lambda_k, lambda_k, lambda_k],
    'error_bound': [error_bound, error_bound, error_bound, error_bound],
    'improvement_vs_raw': [improvement_vs_raw, improvement_vs_raw, improvement_vs_raw, improvement_vs_raw],
    'margin_value': [margin_value, margin_value, margin_value, margin_value],
    'temperature_factor': [temperature_factor, temperature_factor, temperature_factor, temperature_factor]
})

# Compute subset scores
disruption_score = np.zeros(affinity_matrix.shape[0])
for i in range(affinity_matrix.shape[0]):
    for j in range(i + 1, affinity_matrix.shape[0]):
        if group_labels[i] == group_labels[j]:
            disruption_score[i] += 1
            disruption_score[j] += 1
        elif i in selected_indices or j in selected_indices:
            disruption_score[i] += 0.5
            disruption_score[j] += 0.5

subset_scores = pd.DataFrame({
    'item_id': [f'item_{i:04d}' for i in range(affinity_matrix.shape[0])],
    'disruption_score': disruption_score,
    'selected': [1 if i in selected_indices else 0 for i in range(affinity_matrix.shape[0])]
})

# Save the outputs
group_labels.to_csv('results/group_labels.csv', index=False)
subset_scores.to_csv('results/subset_scores.csv', index=False)
pair_regime_summary.to_csv('results/pair_regime_summary.csv', index=False)
bound_comparison.to_csv('results/bound_comparison.csv', index=False)
spectral_report.to_csv('results/spectral_report.csv', index=False)
np.savez('results/adjusted_affinity.npz', raw_normalized=raw_normalized, removed_normalized=removed_normalized, margin_adjusted_normalized=margin_adjusted_normalized, temperature_scaled_normalized=temperature_scaled_normalized, selected_indices=selected_indices, kept_indices=selected_indices)

# Plot the affinity matrix, spectrum, ranked subset scores, and bound comparison
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(affinity_matrix, cmap='viridis')
axs[0, 0].set_title('Affinity Matrix')
axs[0, 1].plot(eigenvalues)
axs[0, 1].set_title('Spectrum')
axs[1, 0].plot(sorted(disruption_score, reverse=True))
axs[1, 0].set_title('Ranked Subset Scores')
axs[1, 1].plot(bound_comparison['error_bound'], label='Error Bound')
axs[1, 1].plot(bound_comparison['improvement_vs_raw'], label='Improvement vs. Raw')
axs[1, 1].legend()
plt.show()
plt.savefig('results/diagnostics.png')