```python
import numpy as np
import json
import os
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm

# Load input data
with open('./data/input_0.json') as f:
    input_0 = json.load(f)
G = input_0['gravitational_constant']
sky_sigma = input_0['sky_sigma']
vlos_sigma = input_0['vlos_sigma']
close_threshold = input_0['close_threshold']
case_order = input_0['case_order']

masses = np.load('./data/input_1.npy')
fit_times = np.load('./data/input_2.npy')
detected_rows = np.load('./data/input_3.npy')
prediction_times = np.load('./data/input_4.npy')
prior_particles = np.load('./data/input_5.npy')

# Function to propagate state
def propagate_state(state, times, masses, G):
    n_bodies = masses.shape[0]
    n_times = len(times)
    positions = np.zeros((n_times, n_bodies, 3))
    velocities = np.zeros((n_times, n_bodies, 3))
    positions[0] = state[:, :3]
    velocities[0] = state[:, 3:]
    
    for i in range(1, n_times):
        dt = times[i] - times[i-1]
        for j in range(n_bodies):
            acc = np.zeros(3)
            for k in range(n_bodies):
                if j != k:
                    r = positions[i-1][j] - positions[i-1][k]
                    r_mag = np.linalg.norm(r)
                    acc += G * masses[k] * r / r_mag**3
            velocities[i][j] = velocities[i-1][j] + acc * dt
        positions[i] = positions[i-1] + velocities[i-1] * dt
    return positions, velocities

# Function to compute likelihood
def compute_likelihood(particles, times, detected_rows, masses, sky_sigma, vlos_sigma):
    n_particles = particles.shape[0]
    n_bodies = masses.shape[0]
    n_times = len(times)
    n_detections = detected_rows.shape[1]
    
    likelihoods = np.zeros((n_particles, n_times))
    
    for p in range(n_particles):
        positions, velocities = propagate_state(particles[p], times, masses, G)
        for t in range(n_times):
            obs = detected_rows[:, t]
            valid_indices = obs[:, 3] == 1
            valid_obs = obs[valid_indices]
            
            if valid_obs.shape[0] == 0:
                continue
            
            # Optimal assignment
            cost_matrix = np.zeros((n_bodies, valid_obs.shape[0]))
            for b in range(n_bodies):
                for i in range(valid_obs.shape[0]):
                    x = valid_obs[i, 0]
                    y = valid_obs[i, 1]
                    vz = valid_obs[i, 2]
                    x_pred = positions[t, b, 0]
                    y_pred = positions[t, b, 1]
                    vz_pred = velocities[t, b, 2]
                    cost_matrix[b, i] = (x - x_pred)**2 / sky_sigma**2 + (y - y_pred)**2 / sky_sigma**2 + (vz - vz_pred)**2 / vlos_sigma**2
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            total_cost = np.sum(cost_matrix[row_ind, col_ind])
            likelihoods[p, t] = np.exp(-total_cost / 2)
    
    return likelihoods

# Function to compute observables
def compute_observables(particles, times, masses):
    n_particles = particles.shape[0]
    n_times = len(times)
    n_bodies = masses.shape[0]
    n_observables = 10
    
    observables = np.zeros((n_particles, n_times, n_observables))
    
    for p in range(n_particles):
        for t in range(n_times):
            positions = particles[p, t, :, :3]
            separations = np.zeros(n_bodies * (n_bodies - 1) // 2)
            idx = 0
            for i in range(n_bodies):
                for j in range(i + 1, n_bodies):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    separations[idx] = np.sqrt(dx**2 + dy**2)
                    idx += 1
            separations.sort()
            observables[p, t, :6] = separations[:6]
            
            vz_values = np.zeros(n_bodies)
            for i in range(n_bodies):
                vz_values[i] = particles[p, t, i, 2]
            vz_values.sort()
            observables[p, t, 6:] = vz_values[:4]
    
    return observables

# Main processing
n_cases = 4
n_particles = prior_particles.shape[1]
n_times = prediction_times.shape[1]
n_obs = 10

# Initialize output arrays
quantiles = np.zeros((n_cases, n_times, n_obs, 3))
risk_summary = np.zeros((n_cases, 9))

for case_id in range(n_cases):
    # Get case-specific data
    masses_case = masses[case_id]
    fit_times_case = fit_times[case_id]
    detected_rows_case = detected_rows[case_id]
    prediction_times_case = prediction_times[case_id]
    prior_particles_case = prior_particles[case_id]
    
    # Step 1: Propagate prior particles to fit times
    positions_fit, velocities_fit = propagate_state(prior_particles_case, fit_times_case, masses_case, G)
    
    # Step 2: Compute likelihoods
    likelihoods = compute_likelihood(prior_particles_case, fit_times_case, detected_rows_case, masses_case, sky_sigma, vlos_sigma)
    
    # Step 3: Normalize weights
    weights = likelihoods / np.sum(likelihoods, axis=0)
    
    # Step 4: Propagate posterior particles to prediction times
    positions_pred = np.zeros((n_particles, n_times, n_bodies, 3))
    velocities_pred = np.zeros((n_particles, n_times, n_bodies, 3))
    
    for p in range(n_particles):
        for t in range(n_times):
            dt = prediction_times_case[t] - prediction_times_case[t-1] if t > 0 else 0
            if t == 0:
                positions_pred[p, t] = prior_particles_case[p, :, :3]
                velocities_pred[p, t] = prior_particles_case[p, :, 3:]
            else:
                for b in range(n_bodies):
                    acc = np.zeros(3)
                    for k in range(n_bodies):
                        if b != k:
                            r = positions_pred[p, t-1, b] - positions_pred[p, t-1, k]
                            r_mag = np.linalg.norm(r)
                            acc += G * masses_case[k] * r / r_mag**3
                    velocities_pred[p, t, b] = velocities_pred[p, t-1, b] + acc * dt
                positions_pred[p, t] = positions_pred[p, t-1] + velocities_pred[p, t-1] * dt
    
    # Step 5: Compute observables
    observables = compute_observables(positions_pred, prediction_times_case, masses_case)
    
    # Step 6: Compute quantiles
    for t in range(n_times):
        for obs in range(n_obs):
            quantiles[case_id, t, obs] = np.percentile(observables[:, t, obs], [5, 50, 95])
    
    # Step 7: Compute risk summary
    min_separations = np.zeros((n_particles, n_times))
    for p in range(n_particles):
        for t in range(n_times):
            positions = positions_pred[p, t]
            separations = np.zeros(n_bodies * (n_bodies - 1) // 2)
            idx = 0
            for i in range(n_bodies):
                for j in range(i + 1, n_bodies):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    separations[idx] = np.sqrt(dx**2 + dy**2)
                    idx += 1
            min_separations[p, t] = np.min(separations)
    
    # Compute probabilities and times
    close_prob = np.mean(weights[np.min(min_separations, axis=1) < close_threshold])
    min_separations_flat = min_separations[np.min(min_separations, axis=1) < close_threshold]
    min_times = prediction_times_case[np.min(min_separations, axis=1) < close_threshold]
    
    p05_time = np.percentile(min_times, 5)
    p50_time = np.percentile(min_times, 50)
    p95_time = np.percentile(min_times, 95)
    
    risk_summary[case_id] = [
        case_id,
        close_prob,
        np.percentile(min_separations_flat, 5),
        np.percentile(min_separations_flat, 50),
        np.percentile(min_separations_flat, 95),
        p05_time,
        p50_time,
        p95_time
    ]

# Save outputs
np.save('prediction_quantiles.npy', quantiles)
np.savetxt('risk_summary.csv', risk_summary, delimiter=',', header='case_id,close_approach_probability,min_projected_separation_p05,min_projected_separation_p50,min_projected_separation_p95,encounter_time_p05,encounter_time_p50,encounter_time_p95', comments='')
```