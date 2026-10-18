```python
import numpy as np
import json
from collections import defaultdict

def make_agent(env_spec, seed):
    H = env_spec['H']
    S = env_spec['S']
    A = env_spec['A']
    reward = env_spec['reward']
    trans = env_spec['trans']
    max_episode = env_spec['max_episode']

    # Initialize Q tables for each step
    td_upper = np.zeros((H+1, S, A))
    td_lower = np.zeros((H+1, S, A))
    ret_upper = np.zeros((H+1, S, A))
    ret_lower = np.zeros((H+1, S, A))
    td_mean = np.zeros((H+1, S, A))
    td_var = np.zeros((H+1, S, A))
    ret_mean = np.zeros((H+1, S, A))
    ret_var = np.zeros((H+1, S, A))

    # Initialize episode buffer
    episode_buffer = []

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    def act(h, state):
        # Compute optimistic Q values
        td_scores = td_upper[h, state, :]
        ret_scores = ret_upper[h, state, :]
        scores = (1 - reward) * td_scores + reward * ret_scores

        # Select action with highest score
        max_score = np.max(scores)
        best_actions = np.where(scores == max_score)[0]

        # Break ties randomly
        return rng.choice(best_actions)

    def observe(h, state, action, reward, next_state, done):
        # Update episode buffer
        episode_buffer.append((h, state, action, reward, next_state, done))

        # Update Q tables
        if not done:
            td_mean[h, state, action] += reward + td_upper[h+1, next_state, :]
            td_var[h, state, action] += (reward + td_upper[h+1, next_state, :] - td_mean[h, state, action]) ** 2
            td_upper[h, state, action] = np.clip(td_mean[h, state, action] + c_td_var * np.sqrt(td_var[h, state, action] / (H + 1)) + c_td_lin * cap * (H + 1) / (H + 1), 0, cap)
            td_lower[h, state, action] = np.clip(td_mean[h, state, action] - c_td_var * np.sqrt(td_var[h, state, action] / (H + 1)) - c_td_lin * cap * (H + 1) / (H + 1), 0, cap)

            ret_mean[h, state, action] += reward + ret_upper[h+1, next_state, :]
            ret_var[h, state, action] += (reward + ret_upper[h+1, next_state, :] - ret_mean[h, state, action]) ** 2
            ret_upper[h, state, action] = np.clip(ret_mean[h, state, action] + c_ret_var * np.sqrt(ret_var[h, state, action] / (H + 1)) + c_ret_lin * cap * (H + 1) / (H + 1), 0, cap)
            ret_lower[h, state, action] = np.clip(ret_mean[h, state, action] - c_ret_var * np.sqrt(ret_var[h, state, action] / (H + 1)) - c_ret_lin * cap * (H + 1) / (H + 1), 0, cap)
        else:
            td_mean[h, state, action] += reward
            td_var[h, state, action] += (reward - td_mean[h, state, action]) ** 2
            td_upper[h, state, action] = np.clip(td_mean[h, state, action] + c_td_var * np.sqrt(td_var[h, state, action] / (H + 1)) + c_td_lin * cap * (H + 1) / (H + 1), 0, cap)
            td_lower[h, state, action] = np.clip(td_mean[h, state, action] - c_td_var * np.sqrt(td_var[h, state, action] / (H + 1)) - c_td_lin * cap * (H + 1) / (H + 1), 0, cap)

            ret_mean[h, state, action] += reward
            ret_var[h, state, action] += (reward - ret_mean[h, state, action]) ** 2
            ret_upper[h, state, action] = np.clip(ret_mean[h, state, action] + c_ret_var * np.sqrt(ret_var[h, state, action] / (H + 1)) + c_ret_lin * cap * (H + 1) / (H + 1), 0, cap)
            ret_lower[h, state, action] = np.clip(ret_mean[h, state, action] - c_ret_var * np.sqrt(ret_var[h, state, action] / (H + 1)) - c_ret_lin * cap * (H + 1) / (H + 1), 0, cap)

        # Clear episode buffer after the episode is done
        if done:
            episode_buffer.clear()

    return act, observe
```