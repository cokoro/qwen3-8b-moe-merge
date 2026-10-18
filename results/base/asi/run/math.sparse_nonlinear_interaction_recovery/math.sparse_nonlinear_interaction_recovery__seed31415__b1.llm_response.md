```python
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from scipy.stats import t
from sklearn.decomposition import PCA
from sklearn.utils import check_array
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings("ignore")

# Load data
design_matrix = np.load('./data/design_matrix.npy')
observations = np.load('./data/observations.npy')
nuisance_covariates = np.load('./data/nuisance_covariates.npy')
environment_id = np.load('./data/environment_id.npy')
group_structure = json.load(open('./data/group_structure.json'))

# Preprocess: adjust for nuisance and environment effects
design_matrix_centered = design_matrix - np.mean(design_matrix, axis=0)
nuisance_effect = np.mean(nuisance_covariates, axis=0)
environment_effect = np.mean(observations[np.unique(environment_id, return_inverse=True)[1]], axis=0)
adjusted_observations = observations - nuisance_effect - environment_effect

# Define candidate transforms
def get_transformed_features(X):
    X_poly = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
    X_centered = X - np.mean(X, axis=0)
    X_centered_quad = X_centered ** 2
    X_arctan = np.arctan(X)
    X_sinusoidal = np.sin(X) + np.cos(X)
    X_radial = np.exp(-X ** 2 / 2)
    X_signed_root = np.sign(X) * np.sqrt(np.abs(X))
    X_transformed = np.hstack([X_poly, X_centered_quad, X_arctan, X_sinusoidal, X_radial, X_signed_root])
    return X_transformed

X_transformed = get_transformed_features(design_matrix)

# Group structure for screening
group_map = np.array([int(g) for g in group_structure['groups']])
group_ids = np.unique(group_map)
group_sizes = [np.sum(group_map == g) for g in group_ids]

# Hyperparameters
sparsity_fraction = 0.1
interaction_density = 0.05
lambda_candidates = np.logspace(-5, 1, 50)
num_folds = 5

# Initialize diagnostics
objective_trace = np.zeros(140)
objective_increments = np.zeros(139)
objective_decrease_count = np.zeros(1)
lambda_cv_trace = np.zeros((len(lambda_candidates), num_folds, 3))
moment_vector = np.zeros(4)
hyperparam_vector = np.array([sparsity_fraction, interaction_density], dtype=np.float32)
rmse_score = np.zeros(1)

# Screening and selection
selected_features = np.zeros(100)
interaction_pairs = np.zeros((100, 100))
interaction_weights = np.zeros((100, 100))
support_indicator = np.zeros(100)
support_score = np.zeros(100)

# Function to compute Huber loss
def huber_loss(y_true, y_pred, delta=1.0):
    residual = y_true - y_pred
    return np.where(np.abs(residual) <= delta, 0.5 * residual ** 2, delta * np.abs(residual) - 0.5 * delta ** 2)

# Function to compute RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to compute objective
def compute_objective(X, y, beta, lambda_val, interaction_pairs, interaction_weights):
    y_pred = X @ beta
    for i in range(100):
        for j in range(i + 1, 100):
            if interaction_pairs[i, j] > 0:
                y_pred += interaction_weights[i, j] * X[i] * X[j]
    loss = huber_loss(y, y_pred, delta=1.0)
    penalty = lambda_val * np.sum(np.abs(beta))
    return np.mean(loss) + penalty

# Function to update parameters
def update_params(X, y, beta, interaction_pairs, interaction_weights, lambda_val, sparsity_fraction, interaction_density):
    # Optimization
    def objective_func(params):
        beta_new = params[:100]
        interaction_weights_new = params[100:]
        y_pred = X @ beta_new
        for i in range(100):
            for j in range(i + 1, 100):
                if interaction_pairs[i, j] > 0:
                    y_pred += interaction_weights_new[i * 100 + j] * X[i] * X[j]
        loss = huber_loss(y, y_pred, delta=1.0)
        penalty = lambda_val * np.sum(np.abs(beta_new))
        return np.mean(loss) + penalty

    # Initial guess
    initial_guess = np.concatenate([beta, np.zeros(100 * 99 // 2)])
    result = minimize(objective_func, initial_guess, method='L-BFGS-B', tol=1e-6)
    beta_new = result.x[:100]
    interaction_weights_new = result.x[100:]
    return beta_new, interaction_weights_new

# Main loop
beta = np.zeros(100)
interaction_weights = np.zeros((100, 100))
for step in range(140):
    # Update parameters
    beta, interaction_weights = update_params(X_transformed, adjusted_observations, beta, interaction_pairs, interaction_weights, lambda_candidates[step], sparsity_fraction, interaction_density)
    
    # Update diagnostics
    objective_trace[step] = compute_objective(X_transformed, adjusted_observations, beta, lambda_candidates[step], interaction_pairs, interaction_weights)
    objective_increments[step] = objective_trace[step] - objective_trace[step - 1] if step > 0 else 0
    objective_decrease_count[0] += np.sum(objective_increments[step] < 0)
    
    # Update support and interaction scores
    support_indicator = np.abs(beta) > 0.1
    support_score = np.abs(beta) / np.max(np.abs(beta))
    interaction_score = np.abs(interaction_weights) / np.max(np.abs(interaction_weights))
    interaction_weight = interaction_weights

# Save outputs
np.save('beta_hat.npy', beta)
np.save('support_indicator.npy', support_indicator)
np.save('support_score.npy', support_score)
np.save('interaction_score.npy', interaction_score)
np.save('interaction_weight.npy', interaction_weight)
np.save('hyperparam_vector.npy', hyperparam_vector)
np.save('objective_trace.npy', objective_trace)
np.save('objective_increments.npy', objective_increments)
np.save('objective_decrease_count.npy', objective_decrease_count)
np.save('rmse_score.npy', rmse(adjusted_observations, X_transformed @ beta))
np.save('moment_vector.npy', np.array([rmse(adjusted_observations, X_transformed @ beta), 100, 0, 0]))

# Save CSVs
beta_hat_df = pd.DataFrame({'feature_index': np.arange(100), 'beta_hat': beta, 'is_selected': support_indicator})
beta_hat_df.to_csv('beta_hat.csv', index=False)

diagnostic_df = pd.DataFrame({
    'step': np.arange(140),
    'objective_value': objective_trace,
    'delta_effect_l2': np.zeros(140),
    'delta_param_inf': np.zeros(140),
    'sparsity_fraction': np.zeros(140),
    'penalty_scale': np.zeros(140)
})
diagnostic_df.to_csv('diagnostic_trace.csv', index=False)

lambda_cv_df = pd.DataFrame({
    'lambda_candidate': lambda_candidates,
    'fold_id': np.zeros(num_folds),
    'validation_huber': np.zeros((len(lambda_candidates), num_folds))
})
lambda_cv_df.to_csv('lambda_cv_trace.csv', index=False)

# Save group structure
with open('group_structure.json', 'w') as f:
    json.dump(group_structure, f)
```