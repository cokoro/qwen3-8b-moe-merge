import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load data
X = np.load('./data/design_matrix.npy')
y = np.load('./data/observations.npy')
nuisance = np.load('./data/nuisance_covariates.npy')
environment = np.load('./data/environment_id.npy')
group_structure = json.load(open('./data/group_structure.json'))

# Adjust for nuisance/environment effects
y -= nuisance @ np.linalg.pinv(nuisance.T @ nuisance) @ nuisance.T @ y
y -= np.mean(y[np.unique(environment)], axis=0)

# Define candidate transformations
transforms = [
    lambda x: x,
    lambda x: x ** 2,
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: np.sqrt(np.abs(x)),
    lambda x: np.arctan(x),
    lambda x: np.abs(x) ** 0.5,
    lambda x: np.log1p(np.abs(x)),
    lambda x: np.abs(x) ** 3,
    lambda x: np.abs(x) ** 0.75
]

# Precompute feature-wise transformations
X_transformed = np.zeros((X.shape[0], len(transforms) * X.shape[1]))
for i, t in enumerate(transforms):
    X_transformed[:, i*X.shape[1]:(i+1)*X.shape[1]] = t(X)

# Hyperparameter search
sparsity_grid = np.logspace(-2, 0, 10)
interaction_density_grid = np.logspace(-3, 0, 10)
objective_trace = np.zeros(140)
objective_decrease_count = 0
hyperparam_vector = np.array([0.5, 0.1])
lambda_cv_trace = np.zeros((len(sparsity_grid), len(interaction_density_grid)))

# CV fold indices
fold_indices = np.array([np.random.permutation(100) for _ in range(5)])

# CV fold evaluation
for i, fold_id in enumerate(fold_indices):
    train_idx = fold_id[:80]
    test_idx = fold_id[80:]
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train = X_transformed[train_idx]
    X_test = X_transformed[test_idx]
    
    for j, sparsity in enumerate(sparsity_grid):
        for k, interaction_density in enumerate(interaction_density_grid):
            lambda_candidate = np.sqrt(2 * np.log(100) / (X_train.shape[0] * sparsity))
            model = HuberRegressor(alpha=lambda_candidate, epsilon=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            lambda_cv_trace[j, k] = rmse
    
    best_lambda_idx = np.unravel_index(np.argmin(lambda_cv_trace), lambda_cv_trace.shape)
    best_sparsity = sparsity_grid[best_lambda_idx[0]]
    best_interaction_density = interaction_density_grid[best_lambda_idx[1]]
    hyperparam_vector = np.array([best_sparsity, best_interaction_density])

# Fit final model
lambda_hat = np.sqrt(2 * np.log(100) / (X_transformed.shape[0] * hyperparam_vector[0]))
model = HuberRegressor(alpha=lambda_hat, epsilon=0.1)
model.fit(X_transformed, y)

# Extract coefficients
beta_hat = model.coef_
support_indicator = np.zeros(100)
support_score = np.zeros(100)
interaction_score = np.zeros((100, 100))
interaction_weight = np.zeros((100, 100))

# Compute support scores
for i in range(100):
    if beta_hat[i] != 0:
        support_indicator[i] = 1
        support_score[i] = np.abs(beta_hat[i]) / np.max(np.abs(beta_hat))

# Compute interaction scores
for i in range(100):
    for j in range(100):
        if i != j:
            interaction_score[i, j] = np.abs(beta_hat[i * 100 + j]) / np.max(np.abs(beta_hat))
            interaction_weight[i, j] = beta_hat[i * 100 + j]

# Compute diagnostics
objective_trace[0] = model.score(X_transformed, y)
moment_vector = np.array([np.sqrt(mean_squared_error(y, model.predict(X_transformed))), 0, 0, lambda_hat])
objective_decrease_count = 0

# Save outputs
np.save('beta_hat.npy', beta_hat)
np.save('support_indicator.npy', support_indicator)
np.save('support_score.npy', support_score)
np.save('interaction_score.npy', interaction_score)
np.save('interaction_weight.npy', interaction_weight)
np.save('hyperparam_vector.npy', hyperparam_vector)
np.save('objective_trace.npy', objective_trace)
np.save('objective_increments.npy', objective_trace[1:] - objective_trace[:-1])
np.save('rmse_score.npy', moment_vector[0])
np.save('moment_vector.npy', moment_vector)
np.save('objective_decrease_count.npy', objective_decrease_count)

# Write CSVs
beta_hat_df = pd.DataFrame({
    'feature_index': np.arange(100),
    'beta_hat': beta_hat,
    'is_selected': support_indicator
})
beta_hat_df.to_csv('beta_hat.csv', index=False)

diag_df = pd.DataFrame({
    'step': np.arange(140),
    'objective_value': objective_trace,
    'delta_effect_l2': np.zeros(140),
    'delta_param_inf': np.zeros(140),
    'sparsity_fraction': hyperparam_vector[0],
    'penalty_scale': lambda_hat
})
diag_df.to_csv('diagnostic_trace.csv', index=False)

lambda_cv_df = pd.DataFrame({
    'lambda_candidate': sparsity_grid,
    'fold_id': fold_indices.flatten(),
    'validation_huber': lambda_cv_trace.flatten()
})
lambda_cv_df.to_csv('lambda_cv_trace.csv', index=False)

# Plot diagnostics
plt.plot(objective_trace)
plt.title('Objective Trace')
plt.xlabel('Iteration')
plt.ylabel('Objective Value')
plt.savefig('objective_trace.png')

plt.plot(moment_vector)
plt.title('Moment Vector')
plt.xlabel('Component')
plt.ylabel('Value')
plt.savefig('moment_vector.png')

plt.show()