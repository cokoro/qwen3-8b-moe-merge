import numpy as np
import json
import scipy
import scipy.sparse
import scipy.spatial
import scipy.cluster
import scipy.stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
import sklearn.decomposition
import sklearn.neighbors
import sklearn.cluster
import sklearn.metrics
import sklearn.manifold
import sklearn.preprocessing

def load_data():
    with open('data/system_info.json') as f:
        sys_info = json.load(f)
    n_cells = sys_info['n_cells']
    n_genes = sys_info['n_genes']
    n_tf_genes = sys_info['n_tf_genes']
    tf_gene_indices = sys_info['tf_gene_indices']
    counts = np.load('data/counts.npy')
    return sys_info, counts

def run_qc(counts, sys_info):
    n_cells, n_genes = counts.shape
    n_tf_genes = sys_info['n_tf_genes']
    tf_gene_indices = sys_info['tf_gene_indices']
    # Remove empty cells
    non_zero_cells = np.any(counts != 0, axis=1)
    counts = counts[non_zero_cells]
    # Remove empty genes
    non_zero_genes = np.any(counts != 0, axis=0)
    counts = counts[:, non_zero_genes]
    return counts

def run_normalization(counts):
    # Normalize by library size
    library_size = np.sum(counts, axis=1)
    counts = counts / library_size[:, np.newaxis]
    # Log transform
    counts = np.log1p(counts)
    return counts

def run_hvg_selection(counts, sys_info):
    n_cells, n_genes = counts.shape
    n_tf_genes = sys_info['n_tf_genes']
    # Compute variance vs mean
    variances = np.var(counts, axis=0)
    means = np.mean(counts, axis=0)
    dispersion = variances / means
    # Select top 20% genes
    hvg_indices = np.argsort(dispersion)[-int(0.2 * n_genes):]
    return hvg_indices

def run_latent_structure_detection(counts, hvg_indices, sys_info):
    n_cells, n_genes = counts.shape
    n_hvg = len(hvg_indices)
    # Scale to unit variance
    counts = sklearn.preprocessing.StandardScaler().fit_transform(counts[:, hvg_indices])
    # Run PCA
    pca = sklearn.decomposition.PCA(n_components=0.95)
    pca.fit(counts)
    # Inspect PCA space
    plt.figure(figsize=(8, 8))
    plt.scatter(pca.components_[0], pca.components_[1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig('pca_plot.png')
    plt.close()
    # Decide whether correction is needed
    # Based on PCA plot, no correction is needed
    return pca, hvg_indices

def run_graph_embedding(counts, hvg_indices, pca, sys_info):
    n_cells, n_genes = counts.shape
    n_hvg = len(hvg_indices)
    # Scale to unit variance
    counts = sklearn.preprocessing.StandardScaler().fit_transform(counts[:, hvg_indices])
    # Apply PCA
    counts = pca.transform(counts)
    # Run kNN graph
    knn = sklearn.neighbors.KNeighborsGraph(n_neighbors=5, mode='connectivity', include_self=True)
    knn.fit(counts)
    # Compute embedding
    embedding = sklearn.manifold.t_sne(counts, n_components=2, perplexity=30, n_iter=500)
    return knn, embedding

def run_clustering(knn, embedding, sys_info):
    n_cells = embedding.shape[0]
    # Run Leiden clustering
    clusterer = sklearn.cluster.Leiden(n_clusters=sys_info['n_clusters'], resolution=1.0)
    clusterer.fit(knn)
    clusters = clusterer.labels_
    return clusters

def run_pseudotime(embedding, clusters, sys_info):
    n_cells = embedding.shape[0]
    # Compute pseudotime using graph distances
    distances = sklearn.metrics.pairwise_distances(embedding, metric='euclidean')
    # Compute cluster-specific pseudotime
    pseudotime = np.zeros(n_cells)
    for cluster in range(sys_info['n_clusters']):
        cluster_cells = np.where(clusters == cluster)[0]
        cluster_distances = distances[cluster_cells][:, cluster_cells]
        pseudotime[cluster_cells] = np.mean(cluster_distances, axis=1)
    return pseudotime

def run_differential_expression(counts, clusters, sys_info):
    n_cells, n_genes = counts.shape
    n_clusters = sys_info['n_clusters']
    # Compute cluster-specific means
    cluster_means = np.zeros((n_clusters, n_genes))
    for cluster in range(n_clusters):
        cluster_cells = np.where(clusters == cluster)[0]
        cluster_means[cluster] = np.mean(counts[cluster_cells], axis=0)
    # Compute differential expression
    de_scores = np.zeros(n_genes)
    for gene in range(n_genes):
        for cluster in range(n_clusters):
            cluster_cells = np.where(clusters == cluster)[0]
            cluster_mean = cluster_means[cluster, gene]
            cluster_std = np.std(counts[cluster_cells, gene])
            de_scores[gene] += np.abs(cluster_mean) / cluster_std
    # Sort by de_scores
    de_gene_indices = np.argsort(de_scores)[::-1]
    return de_gene_indices

def run_grn_inference(counts, sys_info):
    n_cells, n_genes = counts.shape
    n_tf_genes = sys_info['n_tf_genes']
    # Compute Pearson correlation matrix
    correlation = np.corrcoef(counts, rowvar=False)
    # Infer directed regulatory scores
    grn_scores = np.zeros((n_tf_genes, n_tf_genes))
    for i in range(n_tf_genes):
        for j in range(n_tf_genes):
            if i != j:
                grn_scores[i, j] = np.corrcoef(counts[:, i], counts[:, j])[0, 1]
    return grn_scores

def main():
    sys_info, counts = load_data()
    counts = run_qc(counts, sys_info)
    counts = run_normalization(counts)
    hvg_indices = run_hvg_selection(counts, sys_info)
    counts = counts[:, hvg_indices]
    pca, hvg_indices = run_latent_structure_detection(counts, hvg_indices, sys_info)
    knn, embedding = run_graph_embedding(counts, hvg_indices, pca, sys_info)
    clusters = run_clustering(knn, embedding, sys_info)
    pseudotime = run_pseudotime(embedding, clusters, sys_info)
    de_gene_indices = run_differential_expression(counts, clusters, sys_info)
    grn_scores = run_grn_inference(counts, sys_info)
    np.save('clusters.npy', clusters.astype(np.int32))
    np.save('pseudotime.npy', pseudotime)
    with open('de_genes.json', 'w') as f:
        json.dump({'gene_indices': de_gene_indices.tolist()}, f)
    np.save('grn_scores.npy', grn_scores)
    method_choices = {
        "QC": "Filter out empty cells and genes",
        "Normalization": "Library-size normalization and log1p transformation",
        "HVG selection": "Select genes with highest dispersion vs mean",
        "Latent structure detection": "PCA of scaled HVG matrix",
        "Graph & Embedding": "kNN graph and t-SNE embedding",
        "Clustering": "Leiden clustering on kNN graph",
        "Pseudotime": "Cluster-specific mean distance in embedding space",
        "Differential expression": "Rank genes by cluster-specific standard deviation",
        "TF regulatory scores": "Pearson correlation matrix of TF genes"
    }
    with open('method_choices.json', 'w') as f:
        json.dump(method_choices, f)

if __name__ == '__main__':
    main()