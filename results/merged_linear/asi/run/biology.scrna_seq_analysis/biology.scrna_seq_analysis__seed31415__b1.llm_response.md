```python
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsGraph
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.stats import zscore
from scipy.sparse.csgraph import laplacian

# Load data
with open('data/system_info.json') as f:
    sys_info = json.load(f)
n_cells, n_genes, n_tf_genes = int(sys_info['n_cells']), int(sys_info['n_genes']), int(sys_info['n_tf_genes'])
counts = np.load('data/counts.npy').astype(np.float64)

# Step 1: QC / filtering
non_zero_cells = np.any(counts > 0, axis=1)
non_zero_genes = np.any(counts > 0, axis=0)
counts = counts[non_zero_cells][:, non_zero_genes]
n_cells, n_genes = counts.shape
print(f"Filtered to {n_cells} cells and {n_genes} genes")

# Step 2: Normalization & variance stabilization
library_size = np.sum(counts, axis=1)
counts = (counts.T / library_size).T
counts = np.log1p(counts)

# Step 3: Highly variable genes (HVG)
means = np.mean(counts, axis=0)
variances = np.var(counts, axis=0)
hvg_indices = np.argsort(variances)[::-1][:n_tf_genes]
hvg_matrix = counts[:, hvg_indices]

# Step 4: Latent structure detection & correction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(hvg_matrix)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title("PCA of HVGs")
plt.show()

# Check for latent structure
latent_structure = np.corrcoef(pca_result.T)
latent_structure = latent_structure - np.diag(np.diag(latent_structure))
latent_structure = np.sum(np.abs(latent_structure), axis=1)
latent_structure = latent_structure / np.max(latent_structure)
latent_structure = latent_structure > 0.2

if np.any(latent_structure):
    # Correct for latent structure using mutual nearest neighbors
    knn = KNeighborsGraph(n_neighbors=5, mode='connectivity')
    adjacency = knn.fit_transform(hvg_matrix)
    laplacian_matrix = laplacian(adjacency, normed=True)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    correction = eigenvectors[:, np.argsort(eigenvalues)[1:3]]
    hvg_matrix = hvg_matrix @ correction
    hvg_matrix = hvg_matrix / np.linalg.norm(hvg_matrix, axis=1, keepdims=True)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(hvg_matrix)
    plt.scatter(pca_result[:, 0], pca_result[:, 1])
    plt.title("Corrected PCA of HVGs")
    plt.show()

# Step 5: Graph & Embedding
knn = KNeighborsGraph(n_neighbors=10, mode='connectivity')
adjacency = knn.fit_transform(hvg_matrix)
laplacian_matrix = laplacian(adjacency, normed=True)
eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
embedding = eigenvectors[:, np.argsort(eigenvalues)[1:3]]
embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

# Step 6: Clustering
spectral = SpectralClustering(n_clusters=int(sys_info['n_clusters']), affinity='precomputed', n_init=10)
clusters = spectral.fit_predict(pairwise_distances(embedding, metric='euclidean'))

# Step 7: Pseudotime
distances = pairwise_distances(embedding, metric='euclidean')
pseudotime = np.sum(distances, axis=1)

# Step 8: Differential expression
gene_means = np.mean(counts, axis=0)
gene_variances = np.var(counts, axis=0)
gene_z_scores = zscore(gene_means, axis=0)
gene_rankings = np.argsort(gene_z_scores)[::-1]

# Step 9: TF regulatory scores
tf_indices = np.argsort(gene_means)[:n_tf_genes]
tf_counts = counts[:, tf_indices]
tf_corr = np.corrcoef(tf_counts.T)
tf_corr = tf_corr - np.diag(np.diag(tf_corr))
tf_corr = tf_corr / np.max(np.abs(tf_corr))
tf_corr = tf_corr * (np.abs(tf_corr) > 0.2)

# Step 10: Method choices
method_choices = {
    "QC": {"method": "Filter non-zero cells and genes", "params": {}, "reason": "Standard QC step"},
    "Normalization": {"method": "Counts-per-million scaling + log1p", "params": {}, "reason": "Standard normalization"},
    "HVG": {"method": "Variance threshold", "params": {"threshold": "n_tf_genes"}, "reason": "Select TF genes as HVGs"},
    "LatentStructure": {"method": "Mutual nearest neighbors", "params": {"n_neighbors": 5}, "reason": "Detect and correct for latent structure"},
    "Graph": {"method": "KNN graph", "params": {"n_neighbors": 10}, "reason": "Standard kNN graph construction"},
    "Embedding": {"method": "Spectral embedding", "params": {"n_components": 2}, "reason": "Standard spectral embedding"},
    "Clustering": {"method": "Spectral clustering", "params": {"n_clusters": int(sys_info['n_clusters']), "n_init": 10}, "reason": "Standard spectral clustering"},
    "Pseudotime": {"method": "Graph distance sum", "params": {"metric": "euclidean"}, "reason": "Simple pseudotime estimation"},
    "DE": {"method": "Z-score ranking", "params": {"threshold": 0.2}, "reason": "Rank genes by Z-score"},
    "GRN": {"method": "Pearson correlation", "params": {"threshold": 0.2}, "reason": "Correlation-based regulatory inference"}
}

with open('method_choices.json', 'w') as f:
    json.dump(method_choices, f)

np.save('clusters.npy', clusters.astype(np.int32))
np.save('pseudotime.npy', pseudotime.astype(np.float64))
with open('de_genes.json', 'w') as f:
    json.dump({"gene_indices": gene_rankings[:n_tf_genes].tolist()}, f)
np.save('grn_scores.npy', tf_corr.astype(np.float64))
```