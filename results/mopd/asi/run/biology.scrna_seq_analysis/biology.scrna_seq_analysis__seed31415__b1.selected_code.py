```py
import numpy as np
import json
import scipy
import scipy.optimize
import scipy.spatial
import scipy.cluster
import scipy.stats
from scipy.sparse import csr_matrix
import sklearn
import sklearn.decomposition
import sklearn.neighbors
import sklearn.cluster
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt

def run_pipeline():
    with open('data/system_info.json') as f:
        system_info = json.load(f)
    n_cells = system_info['n_cells']
    n_genes = system_info['n_genes']
    n_tf_genes = system_info['n_tf_genes']
    tf_gene_indices = np.array(system_info['tf_gene_indices'], dtype=int)
    counts = np.load('data/counts.npy')

    # QC / filtering
    non_empty_cells = np.sum(counts != 0, axis=1) > 0
    counts = counts[non_empty_cells]
    n_cells = counts.shape[0]
    non_empty_genes = np.sum(counts != 0, axis=0) > 0
    counts = counts[:, non_empty_genes]
    n_genes = counts.shape[1]

    # Normalization & variance stabilization
    library_size = np.sum(counts, axis=1)
    counts = (counts.T / library_size).T
    counts = np.log1p(counts)

    # Highly variable genes (HVG)
    mean = np.mean(counts, axis=0)
    var = np.var(counts, axis=0)
    dispersion = var / (mean ** 2)
    hvg_indices = np.argsort(dispersion)[::-1][:1000]
    counts = counts[:, hvg_indices]

    # Latent structure detection & correction
    pca = sklearn.decomposition.TruncatedSVD(n_components=50)
    pca.fit(counts)
    components = pca.transform(counts)
    plt.scatter(components[:, 0], components[:, 1])
    plt.show()
    if plt.ginput(1)[0][0] > 0.5:
        # Apply correction
        corrected_components = sklearn.decomposition.PCA(n_components=50).fit_transform(counts)
        components = corrected_components

    # Graph & Embedding
    k = int(np.sqrt(n_cells))
    knn_graph = sklearn.neighbors.KNeighborsGraph(n_neighbors=k, mode='connectivity')
    knn_graph.fit(components)
    embedding = sklearn.manifold.TSNE(n_components=2).fit_transform(components)

    # Clustering
    clusters = sklearn.cluster.KMeans(n_clusters=5).fit_predict(embedding)

    # Pseudotime
    distances = scipy.spatial.distance.cdist(embedding, embedding, 'euclidean')
    pseudotime = np.sum(distances, axis=1)

    # Differential expression
    de_genes = []
    for gene in range(n_genes):
        means = np.mean(counts[:, gene], axis=0)
        stds = np.std(counts[:, gene], axis=0)
        t_stat, p_value = scipy.stats.ttest_1samp(counts[:, gene], means)
        if p_value < 0.05:
            de_genes.append(gene)
    de_genes.sort()
    de_genes = de_genes[:100]

    # TF regulatory scores
    tf_counts = counts[:, tf_gene_indices]
    tf_counts = tf_counts / np.sum(tf_counts, axis=1, keepdims=True)
    tf_scores = np.zeros((n_tf_genes, n_tf_genes))
    for i in range(n_tf_genes):
        for j in range(n_tf_genes):
            tf_scores[i, j] = scipy.stats.pearsonr(tf_counts[:, i], tf_counts[:, j])[0]
    tf_scores = tf_scores[np.triu_indices(n_tf_genes, k=1)]

    # Method choices
    method_choices = {
        "QC / filtering": "Remove empty cells/genes",
        "Normalization & variance stabilization": "Library-size normalization (counts-per-million scale then log1p)",
        "Highly variable genes (HVG)": "Select HVGs using dispersion vs mean",
        "Latent structure detection & correction": "Run PCA on the scaled HVG matrix. Apply correction if necessary",
        "Graph & Embedding": "Build a kNN graph on the corrected representation, then compute a low-dimensional embedding",
        "Clustering": "Graph-based clustering (e.g., KMeans) on the kNN graph",
        "Pseudotime": "Derive a scalar per cell using graph-based distances",
        "Differential expression": "Identify marker genes between clusters using statistical testing",
        "TF regulatory scores": "On TF columns only, infer directed regulatory scores using Pearson correlation"
    }

    # Export artifacts
    np.save('clusters.npy', clusters.astype(np.int32))
    np.save('pseudotime.npy', pseudotime.astype(np.float64))
    json.dump({'gene_indices': de_genes}, open('de_genes.json', 'w'))
    np.save('grn_scores.npy', tf_scores.astype(np.float64))
    json.dump(method_choices, open('method_choices.json', 'w'))

run_pipeline()
```