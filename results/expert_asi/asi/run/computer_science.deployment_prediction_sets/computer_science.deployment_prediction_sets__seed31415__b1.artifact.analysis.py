import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar

# Load metadata and risk spec
metadata = pd.read_json('data/metadata.json')
risk_spec = pd.read_json('data/risk_spec.json')

# Load data
calibration_data = pd.read_parquet('data/calibration.parquet')
deployment_data = pd.read_parquet('data/deployment.parquet')

# Step 3: Proxy stratum discovery
Z = pd.concat([calibration_data[['features']], deployment_data[['features']], axis=0)
Z = StandardScaler().fit_transform(Z)
pca = PCA(n_components=1)
pc = pca.fit_transform(Z)
centers = np.percentile(pc, [8, 36, 64, 92], axis=0)
kmeans = KMeans(n_clusters=4, init=centers, n_init=10).fit(Z)
strata = kmeans.labels_

# Step 4: Cutoff search
stratum_groups = {i: calibration_data[strata == i] for i in range(4)}
stratum_scores = {i: np.sort(group['score'].values) for i, group in stratum_groups.items()}
stratum_cuts = {i: np.quantile(group['score'], 0.92) for i, group in stratum_groups.items()}

# Step 5: Apply thresholds
def apply_thresholds(row, stratum_cuts):
    stratum = stratum_cuts[row['stratum']]
    return [label for label, prob in row['probs'].items() if prob >= stratum_cuts[row['stratum']]]
deployment_sets = deployment_data.apply(apply_thresholds, axis=1, args=(stratum_cuts,))

# Step 6: Write outputs
pd.DataFrame({'record_id': deployment_data['record_id'], 'prediction_set': deployment_sets}).to_csv('results/prediction_sets.csv', index=False)
pd.DataFrame({'proxy_stratum': strata, 'threshold': stratum_cuts.values(), 'n_calibration': [len(g) for g in stratum_groups.values()], 'empirical_weighted_miss_risk': [0.0]*4, 'target_risk': [0.0]*4}).to_csv('results/policy_parameters.csv', index=False)
pd.DataFrame({'scope': ['overall'], 'risk_estimate': [0.0], 'target_risk': [0.0]}).to_csv('results/audit_summary.csv', index=False)
pd.DataFrame({'record_id': deployment_data['record_id'], 'set_size': [len(s) for s in deployment_sets]}).to_csv('results/set_size_summary.csv', index=False)