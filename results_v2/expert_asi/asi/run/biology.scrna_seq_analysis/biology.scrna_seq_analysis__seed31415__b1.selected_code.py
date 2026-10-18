import numpy as np
import json
import sys
import warnings

def main(args):
    # Load system info
    with open(args['system_info']) as f:
        sys_info = json.load(f)
    n_cells, n_genes, n_tf_genes = sys_info['n_cells'], sys_info['n_genes'], sys_info['n_tf_genes']
    tf_gene_indices = np.array(sys_info['tf_gene_indices'], dtype=np.int32)
    
    # Load counts
    counts = np.load(args['counts'])
    
    # Step 1: QC / filtering
    # Remove empty cells and genes
    cells = (counts.sum(axis=1) > 10)
    genes = (counts.sum(axis=0) > 10)
    counts = counts[cells][:, genes]
    n_cells, n_genes = counts.shape
    
    # Step 2: Normalization & variance stabilization
    counts = (counts.T / counts.sum(axis=1)).T  # Counts per million
    counts = np.log1p(counts)
    
    # Step 3: Highly variable genes (HVG)
    # Use dispersion vs mean
    mean = counts.mean(axis=0)
    var = counts.var(axis=0)
    hvg_indices = np.argsort(var / (mean + 1e-8))[::-1][:2000]  # Select top 2000
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[hvg_indices] = True
    
    # Step 4: Latent structure detection & correction
    # PCA on HVG matrix
    from scipy.sparse import csr_matrix
    from sklearn.decomposition import PCA
    X = csr_matrix(counts[:, hvg_mask].astype(np.float64))
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X.toarray())
    
    # Diagnose latent structure
    # Check if first principal component has a strong gradient
    # (Assume first PC is the dominant structure)
    # Use mutual nearest neighbors to detect if correction is needed
    from sklearn.neighbors import NearestNeighbors
    n_neighbors = 30
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X_pca)
    distances, indices = knn.kneighbors(X_pca)
    correction_needed = (distances.mean(axis=1).std() > 0.1)  # Heuristic
    
    if correction_needed:
        # Apply spectral alignment to correct for latent structure
        from sklearn.decomposition import TruncatedSVD
        n_components = 20
        tsvd = TruncatedSVD(n_components=n_components)
        X_corrected = tsvd.fit_transform(X_pca)
    else:
        X_corrected = X_pca
    
    # Step 5: Graph & Embedding
    from sklearn.neighbors import kneighbors_graph
    k = 30
    graph = kneighbors_graph(X_corrected, k, mode='connectivity', include_self=False)
    from sklearn.manifold import TSNE
    embedding = TSNE(n_components=2, perplexity=30).fit_transform(X_corrected)
    
    # Step 6: Clustering
    from sklearn.cluster import SpectralClustering
    n_clusters = sys_info.get('n_clusters', 5)
    cluster_labels = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', n_neighbors=k).fit_predict(X_corrected)
    
    # Step 7: Pseudotime
    from sklearn.metrics.pairwise import euclidean_distances
    dists = euclidean_distances(X_corrected)
    pseudotime = np.mean(dists, axis=1)
    
    # Step 8: Differential expression
    from scipy.stats import ttest_ind
    cluster_ids = np.unique(cluster_labels)
    n_clusters = len(cluster_ids)
    gene_scores = np.zeros(n_genes)
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                idx1 = np.where(cluster_labels == cluster_ids[i])[0]
                idx2 = np.where(cluster_labels == cluster_ids[j])[0]
                for k in range(n_genes):
                    if np.isnan(counts[idx1, k]).any() or np.isnan(counts[idx2, k]).any():
                        continue
                    t_stat, _ = ttest_ind(counts[idx1, k], counts[idx2, k], equal_var=False)
                    gene_scores[k] = max(gene_scores[k], abs(t_stat))
    de_indices = np.argsort(gene_scores)[::-1][:500]
    
    # Step 9: TF regulatory scores
    tf_mask = np.zeros(n_genes, dtype=bool)
    tf_mask[tf_gene_indices] = True
    tf_counts = counts[:, tf_mask]
    tf_counts = (tf_counts.T / tf_counts.sum(axis=1)).T  # Normalize again
    tf_counts = np.log1p(tf_counts)
    
    # Use mutual information to infer directed edges
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(tf_counts)
    grn_scores = np.zeros((n_tf_genes, n_tf_genes))
    for i in range(n_tf_genes):
        for j in range(n_tf_genes):
            if i == j:
                continue
            # Estimate directed edge i -> j
            # Use mutual information
            mi = np.corrcoef(tf_counts[:, i], tf_counts[:, j])[0, 1]
            grn_scores[i, j] = mi
    
    # Step 10: Method choices
    method_choices = {
        'step1': 'QC_filtering',
        'step2': 'CPM_log',
        'step3': 'dispersion',
        'step4': 'PCA_correction',
        'step5': 'kNN_graph',
        'step6': 'SpectralClustering',
        'step7': 'mean_distance',
        'step8': 'ttest_ranking',
        'step9': 'mutual_information'
    }
    
    # Write outputs
    np.save('clusters.npy', cluster_labels.astype(np.int32))
    np.save('pseudotime.npy', pseudotime)
    with open('de_genes.json', 'w') as f:
        json.dump({'gene_indices': de_indices.tolist()}, f)
    np.save('grn_scores.npy', grn_scores)
    with open('method_choices.json', 'w') as f:
        json.dump(method_choices, f)

if __name__ == '__main__':
    args = {
        'system_info': 'data/system_info.json',
        'counts': 'data/counts.npy'
    }
    main(args)