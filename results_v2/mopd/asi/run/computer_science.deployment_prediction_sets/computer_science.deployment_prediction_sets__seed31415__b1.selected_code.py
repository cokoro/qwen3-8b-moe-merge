import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Read metadata and risk specification
with open('data/metadata.json') as f:
    metadata = json.load(f)
with open('data/risk_spec.json') as f:
    risk_spec = json.load(f)

# Step 2: Load calibration and deployment data
calibration = pd.read_csv('data/calibration.csv')
deployment = pd.read_csv('data/deployment.csv')
calibration_prob = calibration[['record_id'] + [col for col in calibration.columns if col.startswith('prob_')]]
calibration_labels = calibration[['record_id', 'label']]
deployment_prob = deployment[['record_id'] + [col for col in deployment.columns if col.startswith('prob_')]]
deployment_features = deployment[['record_id'] + [col for col in deployment.columns if col.startswith('feature_')]]

# Step 3: Discover proxy strata
Z = pd.concat([calibration[['feature_0', 'feature_1', 'feature_2', 'feature_3']],
               deployment[['feature_0', 'feature_1', 'feature_2', 'feature_3']]])
Z = Z.values
scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

pca = PCA(n_components=1)
Z_pca = pca.fit_transform(Z_scaled)
Z_pca = Z_pca.flatten()
thresholds = np.percentile(Z_pca, [8, 36, 64, 92])
initial_centers = []
for t in thresholds:
    idx = np.argmin(np.abs(Z_pca - t))
    initial_centers.append(Z_scaled[idx])

kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=10)
clusters = kmeans.fit_predict(Z_scaled)
clusters = pd.Series(clusters, index=Z_pca.index).sort_values().values
stratum_centers = kmeans.cluster_centers_
stratum_centers = stratum_centers[np.argsort(np.dot(stratum_centers, pca.components_.flatten()))]

stratum_labels = [f'stratum_{i}' for i in range(4)]
stratum_assignments = pd.DataFrame({'record_id': deployment['record_id'], 'stratum': clusters})
stratum_assignments.to_csv('results/stratum_assignments.csv', index=False)

# Step 4: Search cutoffs within each proxy stratum
stratum_calibration = []
for stratum in stratum_labels:
    stratum_data = calibration.merge(stratum_assignments[stratum_assignments['stratum'] == stratum], on='record_id')
    stratum_calibration.append(stratum_data)

# Step 5: Calculate risk-weighted miss loss
risk_weights = risk_spec['label_cost']
risk_weights = pd.Series(risk_weights, index=[f'label_{i}' for i in range(10)])
risk_weights = risk_weights.values

# Step 6: Select cutoffs within each proxy stratum
stratum_cuts = []
for stratum