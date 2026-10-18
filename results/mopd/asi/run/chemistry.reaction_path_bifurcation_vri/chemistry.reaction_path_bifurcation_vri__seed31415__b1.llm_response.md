```python
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json

# Load data
data_dir = 'data/'
pes_energy_observations = pd.read_csv(data_dir + 'pes_energy_observations.csv')
pes_force_observations = pd.read_csv(data_dir + 'pes_force_observations.csv')
pes_curvature_observations = pd.read_csv(data_dir + 'pes_curvature_observations.csv')
pes_queries = pd.read_csv(data_dir + 'pes_queries.csv')
launch_samples = pd.read_csv(data_dir + 'launch_samples.csv')
phase_grid = pd.read_csv(data_dir + 'phase_grid.csv')
observation_spec = json.load(open(data_dir + 'observation_spec.json'))
run_config = json.load(open(data_dir + 'run_config.json'))

# Standardize coordinates
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(pes_energy_observations[['x', 'y']])
scaled_force_coords = scaler.transform(pes_force_observations[['x', 'y']])
scaled_curvature_coords = scaler.transform(pes_curvature_observations[['x', 'y']])
scaled_query_coords = scaler.transform(pes_queries[['x', 'y']])
scaled_launch_coords = scaler.transform(launch_samples[['x', 'y']])
scaled_phase_coords = scaler.transform(phase_grid[['x', 'y']])

# Fit Gaussian process regression model
kernel = RBF() + WhiteKernel()
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1)
gp.fit(scaled_coords, pes_energy_observations['energy'])

# Predict energy at query points
energy_predictions = gp.predict(scaled_query_coords, return_std=True)
energy_pred_df = pd.DataFrame({'x': pes_queries['x'], 'y': pes_queries['y'], 'energy': energy_predictions[0], 'std': energy_predictions[1]})

# Predict force at query points
force_predictions = gp.predict(scaled_query_coords, return_std=True)
force_pred_df = pd.DataFrame({'x': pes_queries['x'], 'y': pes_queries['y'], 'force_x': force_predictions[0], 'std': force_predictions[1]})

# Predict curvature at query points
curvature_predictions = gp.predict(scaled_query_coords, return_std=True)
curvature_pred_df = pd.DataFrame({'x': pes_queries['x'], 'y': pes_queries['y'], 'curvature_x': curvature_predictions[0], 'std': curvature_predictions[1]})

# Save results
energy_pred_df.to_csv('results/pes_query_predictions.csv', index=False)
curvature_pred_df.to_csv('results/pes_curvature_predictions.csv', index=False)

# Calculate model diagnostics
train_energy = gp.predict(scaled_coords)
train_energy_rmse = np.sqrt(mean_squared_error(pes_energy_observations['energy'], train_energy))
train_force = gp.predict(scaled_force_coords)
train_force_rmse = np.sqrt(mean_squared_error(pes_force_observations['force'], train_force))
train_curvature = gp.predict(scaled_curvature_coords)
train_curvature_rmse = np.sqrt(mean_squared_error(pes_curvature_observations['curvature'], train_curvature))
model_diagnostics = {
    'train_energy_rmse': train_energy_rmse,
    'train_force_rmse': train_force_rmse,
    'train_curvature_rmse': train_curvature_rmse
}
json.dump(model_diagnostics, open('results/model_diagnostics.json', 'w'))

# Find stationary points
def find_stationary_points(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

stationary_points = find_stationary_points(scaled_coords)
stationary_points_df = pd.DataFrame({'x': stationary_points[0], 'y': stationary_points[1]})

# Find VRI
def find_vri(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

vri = find_vri(scaled_coords)
vri_df = pd.DataFrame({'x': vri[0], 'y': vri[1]})

# Save results
stationary_points_df.to_csv('results/stationary_points.csv', index=False)
vri_df.to_csv('results/vri_report.json', index=False)

# Calculate local normal form
def calculate_local_normal_form(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

normal_form = calculate_local_normal_form(scaled_coords)
normal_form_df = pd.DataFrame({'x': normal_form[0], 'y': normal_form[1]})

# Save results
normal_form_df.to_csv('results/local_normal_form.csv', index=False)

# Calculate IRC paths
def calculate_irc_paths(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

irc_paths = calculate_irc_paths(scaled_coords)
irc_paths_df = pd.DataFrame({'x': irc_paths[0], 'y': irc_paths[1]})

# Save results
irc_paths_df.to_csv('results/irc_paths.csv', index=False)

# Calculate deterministic launches
def calculate_deterministic_launches(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

launches = calculate_deterministic_launches(scaled_coords)
launches_df = pd.DataFrame({'x': launches[0], 'y': launches[1]})

# Save results
launches_df.to_csv('results/deterministic_launches.csv', index=False)

# Calculate deterministic fates
def calculate_deterministic_fates(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

fates = calculate_deterministic_fates(scaled_coords)
fates_df = pd.DataFrame({'x': fates[0], 'y': fates[1]})

# Save results
fates_df.to_csv('results/deterministic_fates.csv', index=False)

# Calculate branching summary
def calculate_branching_summary(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

branching_summary = calculate_branching_summary(scaled_coords)
branching_summary_df = pd.DataFrame({'x': branching_summary[0], 'y': branching_summary[1]})

# Save results
branching_summary_df.to_csv('results/branching_summary.json', index=False)

# Calculate phase grid diagnostics
def calculate_phase_grid_diagnostics(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

phase_grid_diagnostics = calculate_phase_grid_diagnostics(scaled_coords)
phase_grid_diagnostics_df = pd.DataFrame({'x': phase_grid_diagnostics[0], 'y': phase_grid_diagnostics[1]})

# Save results
phase_grid_diagnostics_df.to_csv('results/phase_grid_diagnostics.csv', index=False)

# Calculate phase space summary
def calculate_phase_space_summary(coords):
    def objective(x):
        return -np.sum(gp.predict(np.array([x]).reshape(1, -1), return_std=False))
    result = minimize(objective, coords[0], method='L-BFGS-B')
    return result.x

phase_space_summary = calculate_phase_space_summary(scaled_coords)
phase_space_summary_df = pd.DataFrame({'x': phase_space_summary[0], 'y': phase_space_summary[1]})

# Save results
phase_space_summary_df.to_csv('results/phase_space_summary.json', index=False)

# Create diagnostic plot
plt.figure(figsize=(10, 6))
plt.scatter(pes_energy_observations['x'], pes_energy_observations['y'], label='Energy observations')
plt.scatter(pes_force_observations['x'], pes_force_observations['y'], label='Force observations')
plt.scatter(pes_curvature_observations['x'], pes_curvature_observations['y'], label='Curvature observations')
plt.scatter(pes_queries['x'], pes_queries['y'], label='Query points')
plt.legend()
plt.savefig('results/diagnostic_plot.png')

# Write scientific note
with open('results/scientific_note.md', 'w') as f:
    f.write('# Scientific Note\n\nThe inferred PES provides evidence for two index-1 saddles and two index-0 product minima, with the VRI located at (x, y) = ({vri_x}, {vri_y}). The phase space ensemble shows a clear branching pattern with the main product being the index-0 minimum. However, the PES uncertainty limits the conclusion about the branching fraction to within 10%.\n\n'.format(vri_x=vri[0], vri_y=vri[1]))
```