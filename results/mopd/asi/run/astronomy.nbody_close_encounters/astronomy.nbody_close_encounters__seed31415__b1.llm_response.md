```python
import numpy as np
import json
import scipy.optimize
import scipy.interpolate
import scipy.stats

# Load inputs
gravitational_constant = json.load(open('./data/input_0.json'))['gravitational_constant']
masses = np.load('./data/input_1.npy')
fit_times = np.load('./data/input_2.npy')
detected_rows = np.load('./data/input_3.npy')
prediction_times = np.load('./data/input_4.npy')
prior_particles = np.load('./data/input_5.npy')
sky_sigma = json.load(open('./data/input_0.json'))['sky_sigma']
vlos_sigma = json.load(open('./data/input_0.json'))['vlos_sigma']
close_threshold = json.load(open('./data/input_0.json'))['close_threshold']
case_order = json.load(open('./data/input_0.json'))['case_order']

def compute_observable(particle):
    # Compute sky-plane pair separations and line-of-sight velocities
    x = particle[..., 0]
    y = particle[..., 1]
    vx = particle[..., 3]
    vy = particle[..., 4]
    vz = particle[..., 5]
    
    # Compute pair separations
    d = np.sqrt((x[:, np.newaxis] - x[np.newaxis, :])**2 + (y[:, np.newaxis] - y[np.newaxis, :])**2)
    d = np.sort(d, axis=1)
    
    # Compute line-of-sight velocities
    v = np.sort(vz, axis=1)
    
    return np.concatenate([d, v], axis=1)

def propagate_particle(particle, times):
    # Propagate to prediction times
    x = particle[..., 0]
    y = particle[..., 1]
    vx = particle[..., 3]
    vy = particle[..., 4]
    vz = particle[..., 5]
    
    # Propagate to each time
    for t in times:
        # Update positions
        x += vx * t
        y += vy * t
        # Update velocities
        vx += gravitational_constant * masses[1] / x**2
        vy += gravitational_constant * masses[1] / y**2
        vz += gravitational_constant * masses[1] / x**2
    
    return np.stack([x, y, vx, vy, vz], axis=-1)

def compute_likelihood(particle, times, observations):
    # Propagate to times and compute likelihood
    x = particle[..., 0]
    y = particle[..., 1]
    vx = particle[..., 3]
    vy = particle[..., 4]
    vz = particle[..., 5]
    
    # Compute likelihood at each time
    likelihood = 1.0
    for t in times:
        # Update positions
        x += vx * t
        y += vy * t
        # Update velocities
        vx += gravitational_constant * masses[1] / x**2
        vy += gravitational_constant * masses[1] / y**2
        vz += gravitational_constant * masses[1] / x**2
        
        # Compute likelihood
        for obs in observations:
            if obs[3] == 1:
                # Compute distance
                dx = x - obs[0]
                dy = y - obs[1]
                dv = vz - obs[2]
                # Compute likelihood
                likelihood *= np.exp(-0.5 * (dx**2 / sky_sigma**2 + dy**2 / sky_sigma**2 + dv**2 / vlos_sigma**2))
    
    return likelihood

def compute_risk_summary(particles, times):
    # Compute risk summary
    min_separations = np.full((len(particles), len(times)), np.inf)
    min_times = np.full((len(particles), len(times)), np.inf)
    for i in range(len(particles)):
        for j in range(len(times)):
            # Compute sky-plane pair separations
            d = np.sqrt((particles[i, j, 0] - particles[i, j, 2])**2 + (particles[i, j, 1] - particles[i, j, 3])**2)
            if d < min_separations[i]:
                min_separations[i] = d
                min_times[i] = times[j]
    # Compute quantiles
    p05 = np.percentile(min_separations, 5)
    p50 = np.percentile(min_separations, 50)
    p95 = np.percentile(min_separations, 95)
    # Compute probabilities
    close_prob = np.sum(min_separations < close_threshold) / len(particles)
    return np.array([p05, p50, p95]), np.array([min_times[p05 < close_threshold], min_times[p50 < close_threshold], min_times[p95 < close_threshold]])

# Compute posterior weights
posterior_weights = np.zeros((4, 72))
for i in range(4):
    # Compute likelihood for each particle
    likelihoods = np.zeros((72,))
    for j in range(72):
        # Compute likelihood
        likelihoods[j] = compute_likelihood(prior_particles[i, j], fit_times[i], detected_rows[i])
    # Normalize
    posterior_weights[i] = likelihoods / np.sum(likelihoods)
    
# Compute posterior particles
posterior_particles = np.zeros((4, 72, 4, 6))
for i in range(4):
    for j in range(72):
        posterior_particles[i, j] = propagate_particle(prior_particles[i, j], prediction_times[i])
        
# Compute observables
observables = np.zeros((4, 121, 10, 3))
for i in range(4):
    for j in range(121):
        # Compute observables
        observables[i, j] = compute_observable(posterior_particles[i, j])
        
# Compute quantiles
quantiles = np.zeros((4, 121, 10, 3))
for i in range(4):
    for j in range(121):
        # Compute quantiles
        quantiles[i, j] = np.percentile(observables[i, j], [5, 50, 95], axis=0)

# Save quantiles
np.save('prediction_quantiles.npy', quantiles)

# Compute risk summary
risk_summary = np.zeros((4, 8))
for i in range(4):
    # Compute risk summary
    risk_summary[i] = compute_risk_summary(posterior_particles[i], prediction_times[i])

# Save risk summary
np.savetxt('risk_summary.csv', risk_summary, delimiter=',', fmt='%.6f')
```