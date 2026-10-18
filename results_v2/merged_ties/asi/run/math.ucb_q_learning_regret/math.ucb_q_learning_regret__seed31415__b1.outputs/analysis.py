import numpy as np
import json
from collections import defaultdict

def make_agent(env_spec, seed):
    H, S, A, _ = env_spec["problem_setup"]["dim"]
    rng = np.random.default_rng(seed)

    # One-step estimator
    td_count = np.zeros((H, S, A), dtype=int)
    td_upper = np.zeros((H, S, A))
    td_lower = np.zeros((H, S, A))
    td_mean = np.zeros((H, S, A))
    td_var = np.zeros((H, S, A))

    # Multi-step estimator
    ret_count = np.zeros((H, S, A), dtype=int)
    ret_upper = np.zeros((H, S, A))
    ret_lower = np.zeros((H, S, A))
    ret_mean = np.zeros((H, S, A))
    ret_var = np.zeros((H, S, A))

    # Episode buffer
    episode = []

    def act(h, state):
        if td_count[h][state].sum() == 0:
            return rng.integers(A)
        scores = td_mean[h][state] + w_ret * ret_mean[h][state]
        return rng.choice(np.arange(A), p=1 / np.sum(scores > 0, axis=0))

    def observe(h, state, action, reward, next_state, done):
        td_count[h][state][action] += 1
        td_mean[h][state][action] = reward + td_mean[h+1][next_state][:]
        td_var[h][state][action] = max(td_var[h][state][action] - td_mean[h][state][action]**2, 0)
        n = td_count[h][state][action]
        L = np.log(max(10, 8*S*A*H*1000)) + np.log(max(1, n))
        eta = (H + 1) / (H + n)
        x_up = reward + td_upper[h+1][next_state]
        x_low = reward + td_lower[h+1][next_state]
        td_mean[h][state][action] = eta * x_up + (1 - eta) * td_mean[h][state][action]
        td_lower[h][state][action] = max(td_lower[h][state][action], x_low - c_td_lin * cap * L / n)
        td_upper[h][state][action] = min(td_upper[h][state][action], cap, x_up + c_td_var * np.sqrt(td_var[h][state][action] * L / n) + c_td_lin * cap * L / n)

        episode.append((h, state, action, reward, next_state, done))

        if done:
            G = 0
            for h, state, action, reward, next_state, done in reversed(episode):
                if done:
                    bootstrap = 0
                else:
                    bootstrap = np.max(ret_upper[h+1][next_state][:])
                G = reward + lambda * G + (1 - lambda) * bootstrap
                ret_count[h][state][action] += 1
                ret_mean[h][state][action] = G * ret_count[h][state][action] / ret_count[h][state][action]
                ret_var[h][state][action] = max(ret_var[h][state][action] - ret_mean[h][state][action]**2, 0)
                n = ret_count[h][state][action]
                L = np.log(max(10, 8*S*A*H*1000)) + np.log(max(1, n))
                eta = (H + 1) / (H + n)
                b_ret = c_ret_var * np.sqrt(ret_var[h][state][action] * L / n) + c_ret_lin * cap * L / n
                ret_upper[h][state][action] = np.clip(ret_mean[h][state][action] + b_ret, 0, cap)
                ret_lower[h][state][action] = np.clip(ret_mean[h][state][action] - b_ret, 0, cap)
            episode.clear()

    return act, observe