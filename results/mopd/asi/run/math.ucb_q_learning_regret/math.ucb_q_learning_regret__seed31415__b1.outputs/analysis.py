import numpy as np
import json
import math
import random

def make_agent(env_spec, seed):
    H = env_spec['H']
    S = env_spec['S']
    A = env_spec['A']
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    # Initialize Q tables
    Q_td_up = np.zeros((H, S, A))
    Q_td_low = np.zeros((H, S, A))
    Q_ret_up = np.zeros((H, S, A))
    Q_ret_low = np.zeros((H, S, A))
    # Initialize visit counts
    visit_counts = np.zeros((H, S, A), dtype=int)
    # Initialize moments for TD
    td_first_moments = np.zeros((H, S, A))
    td_second_moments = np.zeros((H, S, A))
    # Initialize moments for return
    ret_first_moments = np.zeros((H, S, A))
    ret_second_moments = np.zeros((H, S, A))
    # Initialize episode buffer
    episode_buffer = []
    # Constants
    lambda_ = 0.60
    c_td_var = 0.01
    c_td_lin = 0.002
    c_ret_var = 0.08
    c_ret_lin = 0.01
    w_ret = 0.35
    
    def act(h, state):
        # Score each action
        scores = np.zeros(A)
        for a in range(A):
            td_score = Q_td_up[h, state, a]
            ret_score = Q_ret_up[h, state, a]
            scores[a] = (1 - w_ret) * td_score + w_ret * ret_score
        # Choose uniformly among maximizers
        max_score = np.max(scores)
        max_actions = np.where(scores == max_score)[0]
        return rng.choice(max_actions)
    
    def observe(h, state, action, reward, next_state, done):
        # Append to episode buffer
        episode_buffer.append((h, state, action, reward, next_state, done))
        if done:
            # Process the episode buffer
            # Start from the terminal state and work backward
            G = 0.0
            for i in reversed(range(len(episode_buffer))):
                h, state, action, reward, next_state, done = episode_buffer[i]
                # Calculate the bootstrap value
                if done:
                    bootstrap = 0.0
                else:
                    bootstrap = np.max(Q_ret_up[h+1, next_state, :])
                # Calculate G
                G = reward + lambda_ * G + (1 - lambda_) * bootstrap
                # Update the return moments for this (h, state, action)
                weight = 1 / (visit_counts[h, state, action] + 1)
                td_first_moments[h, state, action] += weight * G
                td_second_moments[h, state, action] += weight * (G - td_first_moments[h, state, action])**2
                # Update the return Q estimates
                L = math.log(max(10, 8 * S * A * H * 1000)) + math.log(max(1, visit_counts[h, state, action] + 1))
                var_ret = max(td_second_moments[h, state, action] - td_first_moments[h, state, action]**2, 0)
                b_ret = c_ret_var * math.sqrt(var_ret * L / (visit_counts[h, state, action] + 1)) + c_ret_lin * H * L / (visit_counts[h, state, action] + 1)
                Q_ret_up[h, state, action] = np.clip(td_first_moments[h, state, action] + b_ret, 0, H - h + 1)
                Q_ret_low[h, state, action] = np.clip(td_first_moments[h, state, action] - b_ret, 0, H - h + 1)
                # Update the visit count
                visit_counts[h, state, action] += 1
            # Clear the episode buffer
            episode_buffer.clear()
    
    return act, observe