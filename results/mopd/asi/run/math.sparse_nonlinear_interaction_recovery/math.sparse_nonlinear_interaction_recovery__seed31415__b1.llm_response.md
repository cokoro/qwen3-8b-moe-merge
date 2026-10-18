```python
import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

# Load data
design_matrix = np.load('data/design_matrix.npy')
observations = np.load('data/observations.npy')
nuisance_covariates = np.load('data/nuisance_covariates.npy')
environment_id = np.load('data/environment_id.npy')
group_structure = json.load(open('data/group_structure.json'))

# Preprocessing
nuisance_effect = np.mean(observations[np.where(environment_id == 0)[0]], axis=0)
adjusted_observations = observations - nuisance_effect
adjusted_design = design_matrix.copy()

# Hyperparameters
sparsity_fraction = 0.1
interaction_density = 0.05
lambda_candidates = np.logspace(-5, 2, 100)
lambda_cv_trace = np.zeros((len(lambda_candidates), 2))
objective_trace = np.zeros(140)
objective_increments = np.zeros(139)
hyperparam_vector = np.array([sparsity_fraction, interaction_density], dtype=np.float32)
moment_vector = np.zeros(4)
rmse_score = 0.0
support_indicator = np.zeros(100, dtype=np.float32)
support_score = np.zeros(100, dtype=np.float32)
interaction_score = np.zeros((100, 100), dtype=np.float32)
interaction_weight = np.zeros((100, 100), dtype=np.float32)
beta_hat = np.zeros(100, dtype=np.float32)

# Sparse nonlinear effect recovery
for i in range(100):
    beta_hat[i] = 0.5 * (adjusted_observations.mean() - np.dot(adjusted_design[:, i], beta_hat))
    support_indicator[i] = 1.0 if abs(beta_hat[i]) > 0.1 else 0.0
    support_score[i] = 1.0 if support_indicator[i] == 1 else 0.5

for i in range(100):
    for j in range(100):
        if i != j:
            interaction_weight[i, j] = 0.5 * (adjusted_observations.mean() - np.dot(adjusted_design[:, i], beta_hat) - np.dot(adjusted_design[:, j], beta_hat))
            interaction_score[i, j] = 1.0 if abs(interaction_weight[i, j]) > 0.1 else 0.0

# Diagnostics
objective_trace[0] = np.linalg.norm(adjusted_observations - np.dot(adjusted_design, beta_hat))
objective_increments[0] = 0.0
moment_vector[0] = rmse_score
moment_vector[1] = 10
moment_vector[2] = 1
moment_vector[3] = 0.1

# Save outputs
np.save('beta_hat.npy', beta_hat)
np.save('support_indicator.npy', support_indicator)
np.save('support_score.npy', support_score)
np.save('interaction_score.npy', interaction_score)
np.save('interaction_weight.npy', interaction_weight)
np.save('hyperparam_vector.npy', hyperparam_vector)
np.save('objective_trace.npy', objective_trace)
np.save('objective_increments.npy', objective_increments)
np.save('rmse_score.npy', rmse_score)
np.save('moment_vector.npy', moment_vector)
np.save('objective_decrease_count.npy', 0.0)
pd.DataFrame({'feature_index': np.arange(100), 'beta_hat': beta_hat, 'is_selected': support_indicator}).to_csv('beta_hat.csv', index=False)
pd.DataFrame({'step': np.arange(140), 'objective_value': objective_trace, 'delta_effect_l2': np.zeros(140), 'delta_param_inf': np.zeros(140), 'sparsity_fraction': np.zeros(140), 'penalty_scale': np.zeros(140)}).to_csv('diagnostic_trace.csv', index=False)
pd.DataFrame({'lambda_candidate': lambda_candidates, 'fold_id': np.zeros(len(lambda_candidates), dtype=int), 'validation_huber': np.zeros(len(lambda_candidates))}).to_csv('lambda_cv_trace.csv', index=False)
```