import numpy as np
import json

# Read input
json_path = './data/input_0.json'
with open(json_path, 'r') as f:
    params = json.load(f)
G = params['gravitational_constant']
sky_sigma = params['sky_sigma']
vlos_sigma = params['vlos_sigma']
close_threshold = params['close_threshold']
case_order = params['case_order']

masses = np.load('./data/input_1.npy')
fit_times = np.load('./data/input_2.npy')
detected = np.load('./data/input_3.npy')
pred_times = np.load('./data/input_4.npy')
prior_particles = np.load('./data/input_5.npy')

# Step 1: Propagate prior particles to fit times
def propagate_particles(particles, times, masses, G):
    n_bodies = masses.shape[1]
    n_particles = particles.shape[1]
    n_times = times.shape[1]
    n_steps = n_times - 1
    # Initialize outputs
    outputs = np.zeros((n_bodies, n_particles, n_times, 6))
    for i in range(n_bodies):
        body_mass = masses[i, i]
        for j in range(n_particles):
            state = particles[i, j]
            for t in range(n_times):
                if t == 0:
                    outputs[i, j, t] = state
                else:
                    dt = times[t] - times[t-1]
                    # Integrate using Euler method
                    acc = np.zeros(6)
                    for k in range(n_bodies):
                        if k == i:
                            continue
                        dx = outputs[k, j, t-1] - outputs[i, j, t-1]
                        r = np.linalg.norm(dx)
                        acc += -G * masses[k, k] * dx / (r**3)
                    state += acc * dt
                    outputs[i, j, t] = state
    return outputs

# Step 2: Compute likelihoods
def compute_likelihood(particles, times, detected, sky_sigma, vlos_sigma):
    n_bodies = particles.shape[0]
    n_particles = particles.shape[1]
    n_times = times.shape[1]
    n_obs = detected.shape[2]
    # For each time step, compute the likelihood
    log_likelihood = np.zeros((n_bodies, n_particles, n_times))
    for t in range(n_times):
        for b in range(n_bodies):
            x = particles[b, :, t, 0]
            y = particles[b, :, t, 1]
            vz = particles[b, :, t, 5]
            # Match detected rows to bodies
            obs = detected[t, :, b, :2]  # x_sky, y_sky
            v_obs = detected[t, :, b, 2]  # v_los
            # Compute Gaussian likelihood
            for p in range(n_particles):
                x_pred = x[p]
                y_pred = y[p]
                v_pred = vz[p]
                # Compute likelihood for each observation
                for o in range(n_obs):
                    if detected[t, o, b, 3] == 1:
                        dx = x_pred - obs[o, 0]
                        dy = y_pred - obs[o, 1]
                        d = np.sqrt(dx**2 + dy**2)
                        v_err = v_pred - v_obs[o]
                        log_likelihood[b, p, t] += -0.5 * (d**2 / sky_sigma**2 + v_err**2 / vlos_sigma**2)
    return log_likelihood

# Step 3: Normalize weights
def normalize_weights(log_likelihood):
    n_particles = log_likelihood.shape[1]
    weights = np.exp(log_likelihood - np.max(log_likelihood, axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights

# Step 4: Propagate posterior to prediction times
def propagate_posterior(particles, times, weights, pred_times):
    n_bodies = particles.shape[0]
    n_particles = particles.shape[1]
    n_pred = pred_times.shape[1]
    outputs = np.zeros((n_bodies, n_particles, n_pred, 6))
    for i in range(n_bodies):
        for j in range(n_particles):
            state = particles[i, j]
            for t in range(n_pred):
                dt = pred_times[t] - pred_times[t-1] if t > 0 else pred_times[t]
                # Integrate using Euler method
                acc = np.zeros(6)
                for k in range(n_bodies):
                    if k == i:
                        continue
                    dx = outputs[k, j, t-1] - outputs[i, j, t-1]
                    r = np.linalg.norm(dx)
                    acc += -G * masses[k, k] * dx / (r**3)
                state += acc * dt
                outputs[i, j, t] = state
    return outputs

# Step 5: Compute observables
def compute_observables(particles, pred_times, n_obs=10):
    n_bodies = particles.shape[0]
    n_particles = particles.shape[1]
    n_pred = pred_times.shape[1]
    outputs = np.zeros((n_pred, n_obs, 3))
    for t in range(n_pred):
        for p in range(n_particles):
            # Compute pair separations
            sep = []
            for i in range(n_bodies):
                for j in range(i+1, n_bodies):
                    dx = particles[i, p, t, 0] - particles[j, p, t, 0]
                    dy = particles[i, p, t, 1] - particles[j, p, t, 1]
                    sep.append(np.sqrt(dx**2 + dy**2))
            sep.sort()
            sep = sep[:n_obs]
            # Compute velocities
            velocities = []
            for i in range(n_bodies):
                velocities.append(particles[i, p, t, 5])
            velocities.sort()
            velocities = velocities[:n_obs]
            # Aggregate
            for i in range(n_obs):
                outputs[t, i, 0] = sep[i]
                outputs[t, i, 1] = velocities[i]
    return outputs

# Step 6: Save quantiles
def save_quantiles(outputs, pred_times, output_path='prediction_quantiles.npy'):
    n_pred = pred_times.shape[1]
    n_obs = outputs.shape[1]
    quantiles = np.zeros((n_pred, n_obs, 3))
    for t in range(n_pred):
        for i in range(n_obs):
            quantiles[t, i] = np.percentile(outputs[t, i], [5, 50, 95])
    np.save(output_path, quantiles.astype(np.float64))

# Step 7: Compute risk summary
def compute_risk_summary(particles, pred_times, close_threshold, output_path='risk_summary.csv'):
    n_bodies = particles.shape[0]
    n_particles = particles.shape[1]
    n_pred = pred_times.shape[1]
    risk_summary = []
    for case_idx in range(n_bodies):
        case_id = case_order[case_idx]
        # Compute min separation and time
        min_sep = np.inf
        min_time = np.inf
        for p in range(n_particles):
            for t in range(n_pred):
                sep = 0
                for i in range(n_bodies):
                    for j in range(i+1, n_bodies):
                        dx = particles[i, p, t, 0] - particles[j, p, t, 0]
                        dy = particles[i, p, t, 1] - particles[j, p, t, 1]
                        sep = np.sqrt(dx**2 + dy**2)
                        if sep < min_sep:
                            min_sep = sep
                            min_time = pred_times[t]
        # Compute probability
        prob = 0
        for p in range(n_particles):
            for t in range(n_pred):
                sep = 0
                for i in range(n_bodies):
                    for j in range(i+1, n_bodies):
                        dx = particles[i, p, t, 0] - particles[j, p, t, 0]
                        dy = particles[i, p, t, 1] - particles[j, p, t, 1]
                        sep = np.sqrt(dx**2 + dy**2)
                        if sep < close_threshold:
                            prob += weights[p]
                            break
        # Save row
        risk_summary.append({
            'case_id': case_id,
            'close_approach_probability': prob,
            'min_projected_separation_p05': np.percentile(min_sep, 5),
            'min_projected_separation_p50': np.percentile(min_sep, 50),
            'min_projected_separation_p95': np.percentile(min_sep, 95),
            'encounter_time_p05': np.percentile(min_time, 5),
            'encounter_time_p50': np.percentile(min_time, 50),
            'encounter_time_p95': np.percentile(min_time, 95)
        })
    np.savetxt(output_path, [row.values() for row in risk_summary], delimiter=',')

# Main pipeline
prior_particles = np.load('./data/input_5.npy')
fit_times = np.load('./data/input_2.npy')
detected = np.load('./data/input_3.npy')
pred_times = np.load('./data/input_4.npy')

# Step 1: Propagate prior to fit times
particles = propagate_particles(prior_particles, fit_times, masses, G)

# Step 2: Compute likelihoods
log_likelihood = compute_likelihood(particles, fit_times, detected, sky_sigma, vlos_sigma)

# Step 3: Normalize weights
weights = normalize_weights(log_likelihood)

# Step 4: Propagate posterior to prediction times
posterior_particles = propagate_posterior(prior_particles, fit_times, weights, pred_times)

# Step 5: Compute observables
observables = compute_observables(posterior_particles, pred_times)

# Step 6: Save quantiles
save_quantiles(observables, pred_times)

# Step 7: Compute risk summary
compute_risk_summary(posterior_particles, pred_times, close_threshold)