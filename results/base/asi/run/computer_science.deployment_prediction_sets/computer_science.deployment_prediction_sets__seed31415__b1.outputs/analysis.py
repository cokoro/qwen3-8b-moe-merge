import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata and risk specification
metadata_path = 'data/metadata.json'
risk_spec_path = 'data/risk_spec.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)
with open(risk_spec_path, 'r') as f:
    risk_spec = json.load(f)

# Load calibration and deployment data
calibration_data = pd.read_csv('data/calibration.csv')
deployment_data = pd.read_csv('data/deployment.csv')

# Step 2: Load calibration and deployment probability tables, labels, and anonymous features
calibration_features = calibration_data.drop(columns=['record_id', 'label'])
calibration_labels = calibration_data['label']
calibration_probabilities = calibration_data.drop(columns=['record_id', 'label', 'feature_00', 'feature_01', 'feature_02', 'feature_03'])

deployment_features = deployment_data.drop(columns=['record_id', 'label'])
deployment_labels = deployment_data['label']
deployment_probabilities = deployment_data.drop(columns=['record_id', 'label', 'feature_00', 'feature_01', 'feature_02', 'feature_03'])

# Step 3: Discover recoverable proxy strata
Z = pd.concat([calibration_features, deployment_features], axis=0).values
Z_standardized = StandardScaler().fit_transform(Z)

pca = PCA(n_components=1)
pca_result = pca.fit_transform(Z_standardized)
pca_result = pca_result.flatten()

quantiles = np.percentile(pca_result, [8, 36, 64, 92])
initial_centers = []
for q in quantiles:
    idx = np.abs(pca_result - q).argmin()
    initial_centers.append(Z_standardized[idx])

kmeans = KMeans(n_clusters=4, init=initial_centers, n_init=10)
clusters = kmeans.fit_predict(Z_standardized)
cluster_centers = kmeans.cluster_centers_

# Relabel clusters by projection on first principal component
projection = np.dot(cluster_centers, pca.components_.T)
sorted_clusters = np.argsort(projection)
relabelled_clusters = np.zeros_like(clusters)
for i, cluster in enumerate(sorted_clusters):
    relabelled_clusters[clusters == cluster] = i

proxy_strata = np.unique(relabelled_clusters)
proxy_strata = proxy_strata[proxy_strata != 0]  # Remove zero if needed

# Step 4: For each proxy stratum, search cutoffs over a dense probability grid
proxy_strata = proxy_strata[:4]  # Ensure exactly four strata
stratum_cutoffs = {}
for stratum in proxy_strata:
    stratum_calibration = calibration_data[relabelled_clusters == stratum]
    stratum_calibration_probabilities = stratum_calibration.drop(columns=['record_id', 'label'])
    stratum_calibration_labels = stratum_calibration['label']
    
    # Generate probability grid
    prob_grid = np.linspace(0.00, 0.92, 100)
    best_cutoffs = {}
    for label in np.unique(stratum_calibration_labels):
        label_indices = stratum_calibration_labels == label
        label_probabilities = stratum_calibration_probabilities[label_indices]
        label_labels = stratum_calibration_labels[label_indices]
        
        # Find optimal cutoff for this label
        best_cutoff = 0.0
        best_risk = float('inf')
        for cutoff in prob_grid:
            risk = np.mean(label_labels != (stratum_calibration_probabilities >= cutoff))
            if risk < best_risk:
                best_risk = risk
                best_cutoff = cutoff
        best_cutoffs[label] = best_cutoff
    
    stratum_cutoffs[stratum] = best_cutoffs

# Step 5: Compute risk and buffer
risk_budget = risk_spec['target_risk']
confidence_delta = risk_spec['confidence_delta']
label_costs = risk_spec['label_costs']

stratum_parameters = []
for stratum in proxy_strata:
    stratum_calibration = calibration_data[relabelled_clusters == stratum]
    stratum_calibration_probabilities = stratum_calibration.drop(columns=['record_id', 'label'])
    stratum_calibration_labels = stratum_calibration['label']
    
    stratum_n_calibration = len(stratum_calibration)
    stratum_label_costs = [label_costs[label] for label in np.unique(stratum_calibration_labels)]
    stratum_label_costs = np.array(stratum_label_costs)
    
    stratum_weighted_miss_risk = 0.0
    for label in np.unique(stratum_calibration_labels):
        label_indices = stratum_calibration_labels == label
        label_probabilities = stratum_calibration_probabilities[label_indices]
        label_labels = stratum_calibration_labels[label_indices]
        
        cutoff = stratum_cutoffs[stratum][label]
        risk = np.mean(label_labels != (label_probabilities >= cutoff))
        weighted_risk = risk * label_costs[label]
        stratum_weighted_miss_risk += weighted_risk
    
    # Compute buffer
    buffer = confidence_delta * np.sqrt(np.log(1 / stratum_n_calibration) / (2 * stratum_n_calibration))
    stratum_weighted_miss_risk += buffer
    
    stratum_parameters.append({
        'proxy_stratum': stratum,
        'threshold': stratum_cutoffs[stratum],
        'n_calibration': stratum_n_calibration,
        'empirical_weighted_miss_risk': stratum_weighted_miss_risk,
        'target_risk': risk_budget
    })

# Step 6: Apply cutoffs to deployment records
deployment_predictions = []
for index, row in deployment_data.iterrows():
    record_id = row['record_id']
    features = row.drop(columns=['record_id', 'label']).values
    probabilities = row.drop(columns=['record_id', 'label', 'feature_00', 'feature_01', 'feature_02', 'feature_03']).values
    label = row['label']
    
    stratum = relabelled_clusters[np.abs(Z_standardized - features).argmin()]
    stratum = proxy_strata[proxy_strata == stratum][0]
    
    stratum_cutoffs_current = stratum_cutoffs[stratum]
    prediction_set = []
    for label in np.unique(calibration_labels):
        cutoff = stratum_cutoffs_current.get(label, 0.0)
        if probabilities[label] >= cutoff:
            prediction_set.append(label)
    
    if not prediction_set:
        prediction_set.append(label)
    
    deployment_predictions.append({
        'record_id': record_id,
        'prediction_set': ' '.join(prediction_set)
    })

# Step 7: Output files
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Write prediction sets
pd.DataFrame(deployment_predictions).to_csv(os.path.join(results_dir, 'prediction_sets.csv'), index=False)

# Write policy parameters
policy_df = pd.DataFrame(stratum_parameters)
policy_df.to_csv(os.path.join(results_dir, 'policy_parameters.csv'), index=False)

# Write audit summary
audit_summary = pd.DataFrame({
    'scope': ['overall', 'proxy_stratum'],
    'empirical_weighted_miss_risk': [np.mean([p['empirical_weighted_miss_risk'] for p in stratum_parameters]), np.mean([p['empirical_weighted_miss_risk'] for p in stratum_parameters])],
    'target_risk': [risk_budget, risk_budget]
})
audit_summary.to_csv(os.path.join(results_dir, 'audit_summary.csv'), index=False)

# Write set size summary
set_size_summary = pd.DataFrame({
    'record_id': deployment_data['record_id'],
    'set_size': [len(p['prediction_set'].split()) for p in deployment_predictions]
})
set_size_summary.to_csv(os.path.join(results_dir, 'set_size_summary.csv'), index=False)

# Write partition diagnostics
partition_diagnostics = pd.DataFrame({
    'proxy_stratum': proxy_strata,
    'risk_margin': [p['empirical_weighted_miss_risk'] - risk_budget for p in stratum_parameters],
    'set_size': [len(p['prediction_set'].split()) for p in stratum_parameters]
})
partition_diagnostics.to_csv(os.path.join(results_dir, 'partition_diagnostics.csv'), index=False)

# Write distribution diagnostics
distribution_diagnostics = pd.DataFrame({
    'proxy_stratum': proxy_strata,
    'calibration_fraction': [len(calibration_data[relabelled_clusters == stratum]) / len(calibration_data) for stratum in proxy_strata],
    'deployment_fraction': [len(deployment_data[relabelled_clusters == stratum]) / len(deployment_data) for stratum in proxy_strata],
    'density_ratio': [len(deployment_data[relabelled_clusters == stratum]) / len(calibration_data[relabelled_clusters == stratum]) for stratum in proxy_strata]
})
distribution_diagnostics.to_csv(os.path.join(results_dir, 'distribution_diagnostics.csv'), index=False)

# Write calibration diagnostics
calibration_diagnostics = pd.DataFrame({
    'proxy_stratum': proxy_strata,
    'label': np.unique(calibration_labels),
    'score_diagnostics': [np.mean(calibration_data[relabelled_clusters == stratum][label]) for stratum in proxy_strata for label in np.unique(calibration_labels)]
})
calibration_diagnostics.to_csv(os.path.join(results_dir, 'calibration_diagnostics.csv'), index=False)

# Write constraint audit
constraint_audit = pd.DataFrame({
    'proxy_stratum': proxy_strata,
    'empirical_weighted_miss_risk': [p['empirical_weighted_miss_risk'] for p in stratum_parameters],
    'target_risk': [risk_budget for _ in proxy_strata]
})
constraint_audit.to_csv(os.path.join(results_dir, 'constraint_audit.csv'), index=False)

# Generate plots
plt.figure(figsize=(10, 6))
sns.histplot(deployment_data['label'], kde=True)
plt.title('Deployment Label Distribution')
plt.savefig(os.path.join(results_dir, 'set_size_tradeoff.png'))

plt.figure(figsize=(10, 6))
sns.histplot([len(p['prediction_set'].split()) for p in deployment_predictions], kde=True)
plt.title('Prediction Set Size Distribution')
plt.savefig(os.path.join(results_dir, 'partition_audit.png'))

plt.figure(figsize=(10, 6))
sns.histplot([p['empirical_weighted_miss_risk'] for p in stratum_parameters], kde=True)
plt.title('Empirical Weighted Miss Risk Distribution')
plt.savefig(os.path.join(results_dir, 'score_diagnostics.png'))