```bash
import numpy as np
import json

with open('./data/input_0.json') as f:
    config = json.load(f)
    G = config['gravitational_constant']
    sky_sigma = config['sky_sigma']
    vlos_sigma = config['vlos_sigma']
    close_threshold = config['close_threshold']
    case_order = config['case_order']

# Load input data
m = np.load('./data/input_1.npy') # shape (4, 4)
t_fit = np.load('./data/input_2.npy') # shape (4, 32)
d = np.load('./data/input_3.npy') # shape (4, 32, 4, 4)
t_pred = np.load('./data/input_4.npy') # shape (4, 121)
p = np.load('./data/input_5.npy') # shape (4, 72, 4, 6)

# Convert to float64
t_fit = t_fit.astype(np.float64)
t_pred = t_pred.astype(np.float64)

# Compute time step
dt = t_fit[0, 1] - t_fit[0, 0]

# Compute detection mask
detected = (d[..., 3] == 1)
d = d[detected]

# Compute Gaussian likelihoods
log_sigma2 = np.log(sky_sigma**2)
log_vlos2 = np.log(vlos_sigma**2)
log_likelihood = np.zeros((4, 72, 32))
for case in range(4):
    for t in range(32):
        for p_idx in range(72):
            for body in range(4):
                log_likelihood[case, p_idx, t] += -0.5 * ((p[case, p_idx, body, 0] - d[case, t, body, 0]) ** 2 + (p[case, p_idx, body, 1] - d[case, t, body, 1]) ** 2) / sky_sigma ** 2
                log_likelihood[case, p_idx, t] += -0.5 * ((p[case, p_idx, body, 3] - d[case, t, body, 2]) ** 2) / vlos_sigma ** 2

# Normalize weights
log_weights = np.zeros((4, 72, 32))
for case in range(4):
    log_weights[case, :, :] = log_likelihood[case, :, :] - np.max(log_likelihood[case, :, :], axis=1, keepdims=True)
    log_weights[case, :, :] -= np.log(np.sum(np.exp(log_weights[case, :, :]), axis=1, keepdims=True))

# Propagate posterior to prediction times
log_posterior = np.zeros((4, 72, 121))
for case in range(4):
    for p_idx in range(72):
        for t in range(121):
            for body in range(4):
                log_posterior[case, p_idx, t] += -0.5 * ((p[case, p_idx, body, 0] - d[case, t, body, 0]) ** 2 + (p[case, p_idx, body, 1] - d[case, t, body, 1]) ** 2) / sky_sigma ** 2
                log_posterior[case, p_idx, t] += -0.5 * ((p[case, p_idx, body, 3] - d[case, t, body, 2]) ** 2) / vlos_sigma ** 2

# Normalize weights
log_weights_pred = np.zeros((4, 72, 121))
for case in range(4):
    log_weights_pred[case, :, :] = log_posterior[case, :, :] - np.max(log_posterior[case, :, :], axis=1, keepdims=True)
    log_weights_pred[case, :, :] -= np.log(np.sum(np.exp(log_weights_pred[case, :, :]), axis=1, keepdims=True))

# Compute quantiles
quantiles = np.zeros((4, 121, 10, 3))
for case in range(4):
    for t in range(121):
        for obs in range(10):
            for q in range(3):
                sorted_obs = np.sort(np.stack([p[case, p_idx, i, 0] - p[case, p_idx, j, 0] for i in range(4) for j in range(4) if i < j] for p_idx in range(72)), axis=1)
                quantiles[case, t, obs, q] = np.percentile(sorted_obs, [5, 50, 95])[q]

# Write prediction quantiles
np.save('prediction_quantiles.npy', quantiles)

# Compute risk summary
risk = np.zeros((4, 8))
for case in range(4):
    risk[case, 0] = case_order[case]
    risk[case, 1] = np.sum(np.exp(log_weights_pred[case, :, :]) < 1e-4) / 72
    risk[case, 2] = np.min(np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) for p_idx in range(72))
    risk[case, 3] = np.min(np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) for p_idx in range(72))
    risk[case, 4] = np.min(np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) for p_idx in range(72))
    risk[case, 5] = np.min(np.min(t_pred[case, p_idx] for p_idx in range(72) if np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) < close_threshold))
    risk[case, 6] = np.min(np.min(t_pred[case, p_idx] for p_idx in range(72) if np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) < close_threshold))
    risk[case, 7] = np.min(np.min(t_pred[case, p_idx] for p_idx in range(72) if np.min(np.sqrt((p[case, p_idx, i, 0] - p[case, p_idx, j, 0]) ** 2 + (p[case, p_idx, i, 1] - p[case, p_idx, j, 1]) ** 2) for i in range(4) for j in range(4) if i < j) < close_threshold))
np.savetxt('risk_summary.csv', risk, fmt='%d,%d,%f,%f,%f,%f,%f,%f', delimiter=',')

```