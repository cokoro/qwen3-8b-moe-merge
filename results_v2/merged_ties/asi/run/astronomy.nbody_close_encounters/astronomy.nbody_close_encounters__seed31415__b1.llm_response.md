```bash
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

def load_input(path):
    return np.load(path)

def save_output(path, arr):
    np.save(path, arr)

def solve_2d_matching(a, b):
    # a, b: 2D arrays with shape (n, 2)
    # find assignment of rows from a to b that minimizes sum of squared distances
    # returns assignment (a_row_idx -> b_row_idx)
    cost = np.sum((a[:, None] - b[None]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def solve_1d_matching(a, b):
    # a, b: 1D arrays
    # find assignment of rows from a to b that minimizes sum of squared distances
    # returns assignment (a_row_idx -> b_row_idx)
    cost = (a[:, None] - b[None]) ** 2
    row_ind, col_ind = linear_sum_assignment(cost)
    return row_ind, col_ind

def load_data(case_order, input_0_path, input_1_path, input_2_path, input_3_path, input_4_path, input_5_path, sky_sigma, vlos_sigma):
    input_0 = json.load(open(input_0_path))
    input_1 = load_input(input_1_path)
    input_2 = load_input(input_2_path)
    input_3 = load_input(input_3_path)
    input_4 = load_input(input_4_path)
    input_5 = load_input(input_5_path)

    case_params = [input_0[c] for c in case_order]
    G = case_params[0]['gravitational_constant']
    close_threshold = case_params[0]['close_threshold']

    for case_id, case in enumerate(case_params):
        print(f"Processing case {case_id}...")

        # Prepare data for this case
        masses = input_1[case_id]
        times_fit = input_2[case_id]
        detections = input_3[case_id]
        times_pred = input_4[case_id]
        prior_particles = input_5[case_id]

        # Compute the prior likelihood and normalize
        likelihood = np.ones(prior_particles.shape[0])
        for t in times_fit:
            # Get the predicted positions at this time
            pred_pos = prior_particles[:, :, :, :3] @ (prior_particles[:, :, :, 3:6] * t)
            # Get the observed positions at this time
            obs_pos = detections[detections[:, 3] == 1, :3]
            # Match predicted to observed positions
            row_ind, col_ind = solve_2d_matching(pred_pos, obs_pos)
            # Compute log likelihood
            dists = np.sqrt(np.sum((pred_pos[row_ind] - obs_pos[col_ind])**2, axis=1))
            log_like = -0.5 * np.sum(dists**2 / sky_sigma**2, axis=0)
            # Apply Gaussian noise model
            log_like -= np.sum(0.5 * np.log(2*np.pi*sky_sigma**2) + 0.5 * np.log(2*np.pi*vlos_sigma**2), axis=0)
            # Apply likelihood to the prior
            likelihood *= np.exp(log_like)
        # Normalize
        likelihood /= np.sum(likelihood)

        # Propagate the posterior particles forward in time
        # and accumulate the posterior over the prediction times
        posterior_particles = prior_particles.copy()
        posterior_weights = likelihood.copy()
        for t in times_pred:
            # Propagate each particle forward in time
            # and accumulate the posterior over the prediction times
            for i in range(prior_particles.shape[0]):
                if posterior_weights[i] > 0:
                    # Propagate the particle forward in time
                    pred_pos = posterior_particles[i, :, :, :3] @ (posterior_particles[i, :, :, 3:6] * t)
                    posterior_particles[i, :, :, :3] = pred_pos

        # Save the posterior particles
        save_output(f"case{case_id}_posterior.npy", posterior_particles)

        # Compute the posterior quantiles for the 10 observables
        quantiles = np.zeros((len(times_pred), 10, 3))
        for i, t in enumerate(times_pred):
            obs = np.zeros((prior_particles.shape[0], 10))
            for j in range(prior_particles.shape[0]):
                if posterior_weights[j] > 0:
                    # Compute the observable vector
                    obs[j, :6] = np.sort(np.sqrt(np.sum((posterior_particles[j, :, :, :2] - posterior_particles[j, :, :, :2])**2, axis=1)))
                    obs[j, 6:] = np.sort(posterior_particles[j, :, :, 2])
            # Compute the quantiles
            for k in range(10):
                quantiles[i, k, 0] = np.quantile(obs[:, k], 0.05)
                quantiles[i, k, 1] = np.quantile(obs[:, k], 0.5)
                quantiles[i, k, 2] = np.quantile(obs[:, k], 0.95)
        save_output(f"case{case_id}_quantiles.npy", quantiles)

        # Compute the risk summary
        risk_summary = []
        for i, t in enumerate(times_pred):
            # Find the particle with the smallest sky-plane pair separation
            # over the prediction grid
            min_separation = np.inf
            min_time = np.inf
            for j in range(prior_particles.shape[0]):
                if posterior_weights[j] > 0:
                    # Compute the observable vector
                    obs = np.zeros((prior_particles.shape[0], 10))
                    for k in range(prior_particles.shape[0]):
                        if posterior_weights[k] > 0:
                            obs[k, :6] = np.sort(np.sqrt(np.sum((posterior_particles[k, :, :, :2] - posterior_particles[k, :, :, :2])**2, axis=1)))
                            obs[k, 6:] = np.sort(posterior_particles[k, :, :, 2])
                    # Find the particle with the smallest sky-plane pair separation
                    min_separation = np.min(obs[:, :6])
                    min_time = t
                    break
            # Compute the posterior probability that the particle's minimum sky-plane pair separation
            # over the prediction grid is below the close threshold
            close_prob = np.sum(posterior_weights[np.sum(obs[:, :6] < close_threshold, axis=1)] > 0) / np.sum(posterior_weights)
            risk_summary.append({
                'case_id': case_id,
                'close_approach_probability': close_prob,
                'min_projected_separation_p05': np.quantile(obs[:, :6], 0.05),
                'min_projected_separation_p50': np.quantile(obs[:, :6], 0.5),
                'min_projected_separation_p95': np.quantile(obs[:, :6], 0.95),
                'encounter_time_p05': np.quantile(times_pred, 0.05),
                'encounter_time_p50': np.quantile(times_pred, 0.5),
                'encounter_time_p95': np.quantile(times_pred, 0.95),
            })
        # Save the risk summary
        df = pd.DataFrame(risk_summary)
        df.to_csv(f"risk_summary.csv", index=False)

```