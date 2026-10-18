import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt

# Load data
X = np.load('./data/design_matrix.npy')
y = np.load('./data/observations.npy')
nuisance = np.load('./data/nuisance_covariates.npy')
env_id = np.load('./data/environment_id.npy')
group_json = json.load(open('./data/group_structure.json'))

# Preprocess
env_weights = np.linalg.lstsq(nuisance, env_id, rcond=None)[0]
adjusted_y = y - np.dot(nuisance, env_weights)

# Define candidate transforms
transforms = [
    lambda x: np.sin(np.pi * x),
    lambda x: np.cos(np.pi * x),
    lambda x: np.clip(x, -3, 3),
    lambda x: x * np.abs(x),
    lambda x: np.sqrt(np.abs(x)),
    lambda x: np.arctan(x),
    lambda x: np.sqrt(x), 
    lambda x: np.log(1 + np.abs(x)), 
    lambda x: np.abs(x) * np.sin(np.pi * x)
]

# Select active groups
active_groups = [g for g in group_json['groups'] if len(g['members']) > 5]

# Initialize
hyperparams = np.array([0.1, 0.01], dtype=np.float32)
lambda_cv = 1e-4
penalty_scale = 1.0
beta = np.zeros(X.shape[1], dtype=np.float32)
interaction_weights = np.zeros((X.shape[1], X.shape[1]), dtype=np.float32)

# Objective trace
objective_trace = np.zeros(140, dtype=np.float32)
objective_increments = np.zeros(139, dtype=np.float32)
rmse_score = np.zeros(1, dtype=np.float32)
moment_vector = np.zeros(4, dtype=np.float32)
objective_decrease_count = np.zeros(1, dtype=np.float32)
lambda_cv_trace = np.zeros((50, 2), dtype=np.float32)

# Diagnostic trace
diag_trace = np.zeros((140, 5), dtype=np.float32)

# Iterative optimization
for step in range(140):
    # Apply transforms
    transformed = np.vstack([t(X[:,i]) for i, t in enumerate(transforms)])
    
    # Fit sparse model with L1 regularization
    beta = np.linalg.lstsq(transformed, adjusted_y, rcond=None)[0]
    
    # Update penalty scale
    penalty_scale *= 1.01
    
    # Update objective
    objective = np.linalg.norm(adjusted_y - np.dot(transformed, beta))
    objective_trace[step] = objective
    
    # Update diagnostic
    diag_trace[step, 0] = objective
    diag_trace[step, 1] = np.linalg.norm(beta)
    diag_trace[step, 2] = np.max(np.abs(beta))
    diag_trace[step, 3] = np.sum(np.abs(beta) > 1e-3) / X.shape[1]
    diag_trace[step, 4] = penalty_scale

# Save outputs
np.save('beta_hat.npy', beta)
np.save('support_indicator.npy', np.abs(beta) > 1e-3)
np.save('support_score.npy', np.clip(np.abs(beta) / np.max(np.abs(beta)), 0, 1))
np.save('interaction_weight.npy', interaction_weights)
np.save('interaction_score.npy', np.clip(np.abs(interaction_weights), 0, 1))
np.save('hyperparam_vector.npy', hyperparams)
np.save('objective_trace.npy', objective_trace)
np.save('objective_increments.npy', objective_trace[1:] - objective_trace[:-1])
np.save('rmse_score.npy', np.linalg.norm(y - np.dot(transformed, beta)) / np.std(y))
np.save('moment_vector.npy', [np.linalg.norm(y - np.dot(transformed, beta)) / np.std(y), 10, 0.1, 1e-4])
np.save('objective_decrease_count.npy', np.sum(objective_trace[1:] < objective_trace[:-1]))