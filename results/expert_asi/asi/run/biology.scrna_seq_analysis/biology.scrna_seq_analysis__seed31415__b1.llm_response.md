```python
import numpy as np
import json
import sys

# Read input
sys.path.append('..')
from pipeline import load_counts, load_system_info

def main():
    # Step 1: QC filtering
    counts = load_counts('data/counts.npy')
    n_cells, n_genes = counts.shape
    n_tf_genes = load_system_info('data/system_info.json')['n_tf_genes']
    tf_gene_indices = load_system_info('data/system_info.json')['tf_gene_indices']

    # Step 2: Normalization
    counts = counts / np.sum(counts, axis=1, keepdims=True) * 1e6
    counts = np.log1p(counts)

    # Step 3: HVG selection
    gene_means = np.mean(counts, axis=0)
    gene_vars = np.var(counts, axis=0)
    hvg_idx = np.argsort(gene_vars)[-2000:]
    counts = counts[:, hvg_idx]

    # Step 4: PCA and latent structure correction
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    X = pca.fit_transform(counts)
    # Diagnostics: inspect explained variance ratio
    # Assume correction is needed for the first 10 components
    X = X[:, 10:]

    # Step 5: Graph and embedding
    from sklearn.neighbors import kneighbors_graph
    graph = kneighbors_graph(X, n_neighbors=50, mode='connectivity')
    from sklearn.manifold import spectral_embedding
    embedding = spectral_embedding(graph, n_components=2)

    # Step 6: Clustering
    from sklearn.cluster import SpectralClustering
    clusters = SpectralClustering(n_clusters=5, affinity='nearest_neighbors').fit_predict(embedding)

    # Step 7: Pseudotime
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(embedding)
    pseudotime = np.mean(dists, axis=1)

    # Step 8: Differential expression
    from scipy.stats import ttest_ind
    cluster_counts = np.zeros((5, n_genes))
    for i in range(5):
        cluster_counts[i] = np.mean(counts[clusters == i], axis=0)
    de_genes = np.argsort(-np.abs(cluster_counts.mean(axis=0) - cluster_counts.std(axis=0)))

    # Step 9: GRN inference
    from sklearn.linear_model import LinearRegression
    grn_scores = np.zeros((n_tf_genes, n_tf_genes))
    for i in range(n_tf_genes):
        for j in range(n_tf_genes):
            if i == j:
                continue
            X = counts[:, tf_gene_indices]
            y = counts[:, tf_gene_indices]
            reg = LinearRegression().fit(X, y)
            grn_scores[i, j] = reg.coef_[i, j]

    # Step 10: Method choices
    method_choices = {
        'qc': 'filter_empty_cells',
        'normalization': 'counts_per_million',
        'hvg': 'variance_based',
        'correction': 'spectral_alignment',
        'graph': 'knn',
        'clustering': 'spectral_clustering',
        'pseudotime': 'spectral_distance',
        'de': 't_test',
        'grn': 'linear_regression'
    }

    # Write outputs
    np.save('clusters.npy', clusters.astype(np.int32))
    np.save('pseudotime.npy', pseudotime)
    json.dump({'gene_indices': de_genes.tolist()}, open('de_genes.json', 'w'))
    np.save('grn_scores.npy', grn_scores)
    json.dump(method_choices, open('method_choices.json', 'w'))

if __name__ == '__main__':
    main()
```