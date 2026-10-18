```python
import numpy as np
import json
import os

# Read input files
json_file = open('./data/input_0.json')
json_data = json.load(json_file)
json_file.close()

gravitational_constant = json_data['gravitational_constant']
sky_sigma = json_data['sky_sigma']
vlos_sigma = json_data['vlos_sigma']
close_threshold = json_data['close_threshold']
case_order = json_data['case_order']

input_1 = np.load('./data/input_1.npy')  # masses
input_2 = np.load('./data/input_2.npy')  # fit times
input_3 = np.load('./data/input_3.npy')  # shuffled noisy detections
input_4 = np.load('./data/input_4.npy')  # prediction times
input_5 = np.load('./data/input_5.npy')  # prior particles

# Step 1: Propagate prior particles to fit times
prior_particles = input_5
num_bodies = input_1.shape[1]
num_particles = prior_particles.shape[1]
num_fit_times = input_2.shape[1]
num_prediction_times = input_4.shape[1]

# Step 2: At each fit time, compute likelihood for each particle
likelihoods = np.zeros((4, num_particles, num_fit_times))
for case_idx in range(4):
    for t_idx, t in enumerate(input_2[case_idx]):
        for p_idx, particle in enumerate(prior_particles[case_idx]):
            x = particle[:, 0]
            y = particle[:, 1]
            z = particle[:, 2]
            vx = particle[:, 3]
            vy = particle[:, 4]
            vz = particle[:, 5]
            # Propagate to time t
            # Calculate new position and velocity
            dt = t - 0
            x_new = x + vx * dt
            y_new = y + vy * dt
            z_new = z + vz * dt
            vx_new = vx + 0
            vy_new = vy + 0
            vz_new = vz + 0
            # Compute predicted detections
            predicted_detections = np.column_stack((x_new, y_new, vz_new))
            # Compare with observed detections
            observed_detections = input_3[case_idx][t_idx]
            # Only consider detected rows
            detected_rows = observed_detections[observed_detections[:, 3] == 1]
            # Compute likelihood
            likelihood = 1.0
            for obs in detected_rows:
                x_obs = obs[0]
                y_obs = obs[1]
                vlos_obs = obs[2]
                x_diff = x_new - x_obs
                y_diff = y_new - y_obs
                vlos_diff = vz_new - vlos_obs
                likelihood *= np.exp(-0.5 * (x_diff**2 / sky_sigma**2 + y_diff**2 / sky_sigma**2 + vlos_diff**2 / vlos_sigma**2))
            likelihoods[case_idx, p_idx, t_idx] = likelihood

# Step 3: Convert likelihoods to posterior weights
posterior_weights = np.zeros((4, num_particles))
for case_idx in range(4):
    for p_idx in range(num_particles):
        posterior_weights[case_idx, p_idx] = np.prod(likelihoods[case_idx, p_idx, :])

# Step 4: Propagate posterior particles to all prediction times
posterior_particles = np.zeros((4, num_particles, num_prediction_times, 6))
for case_idx in range(4):
    for p_idx in range(num_particles):
        for t_idx, t in enumerate(input_4[case_idx]):
            # Propagate to time t
            dt = t - 0
            # Use the same particle as before
            posterior_particles[case_idx, p_idx, t_idx] = prior_particles[case_idx, p_idx]

# Step 5: Compute label-free observables
observables = np.zeros((4, num_prediction_times, 10))
for case_idx in range(4):
    for t_idx in range(num_prediction_times):
        for p_idx in range(num_particles):
            # Compute sky-plane pair separations
            x = posterior_particles[case_idx, p_idx, t_idx, 0]
            y = posterior_particles[case_idx, p_idx, t_idx, 1]
            z = posterior_particles[case_idx, p_idx, t_idx, 2]
            vx = posterior_particles[case_idx, p_idx, t_idx, 3]
            vy = posterior_particles[case_idx, p_idx, t_idx, 4]
            vz = posterior_particles[case_idx, p_idx, t_idx, 5]
            # Compute pair separations
            separations = np.sqrt((x - x)**2 + (y - y)**2)
            # Sort ascending
            separations = np.sort(separations)
            # Add to observables
            observables[case_idx, t_idx, 0:6] = separations
            observables[case_idx, t_idx, 6:10] = np.sort(vz)

# Step 6: Save posterior weighted quantiles
quantiles = np.zeros((4, num_prediction_times, 10, 3))
for case_idx in range(4):
    for t_idx in range(num_prediction_times):
        for obs_idx in range(10):
            # Get the observables for this case, time, and observable
            obs_values = observables[case_idx, t_idx, obs_idx]
            # Sort the values
            obs_values_sorted = np.sort(obs_values)
            # Compute the quantiles
            quantiles[case_idx, t_idx, obs_idx, 0] = np.quantile(obs_values_sorted, 0.05)
            quantiles[case_idx, t_idx, obs_idx, 1] = np.quantile(obs_values_sorted, 0.5)
            quantiles[case_idx, t_idx, obs_idx, 2] = np.quantile(obs_values_sorted, 0.95)

np.save('prediction_quantiles.npy', quantiles)

# Step 7: Compute risk summary
risk_summary = np.zeros((4, 8))
for case_idx in range(4):
    case_id = case_order[case_idx]
    # Find the minimum sky-plane pair separation over the prediction grid
    min_separations = np.zeros(num_prediction_times)
    for t_idx in range(num_prediction_times):
        min_separations[t_idx] = np.min(observables[case_idx, t_idx, 0:6])
    # Find the time of the minimum separation
    min_separation_time = np.argmin(min_separations)
    # Find the posterior probability of being below the close threshold
    close_approach_prob = np.sum(posterior_weights[case_idx] * (min_separations < close_threshold)) / np.sum(posterior_weights[case_idx])
    # Compute the p05, p50, p95 of the minimum separation
    min_separation_quantiles = np.quantile(min_separations, [0.05, 0.5, 0.95])
    risk_summary[case_idx, 0] = case_id
    risk_summary[case_idx, 1] = close_approach_prob
    risk_summary[case_idx, 2] = min_separation_quantiles[0]
    risk_summary[case_idx, 3] = min_separation_quantiles[1]
    risk_summary[case_idx, 4] = min_separation_quantiles[2]
    risk_summary[case_idx, 5] = min_separation_time
    risk_summary[case_idx, 6] = min_separation_time
    risk_summary[case_idx, 7] = min_separation_time

np.savetxt('risk_summary.csv', risk_summary, fmt='%d', delimiter=',')

```