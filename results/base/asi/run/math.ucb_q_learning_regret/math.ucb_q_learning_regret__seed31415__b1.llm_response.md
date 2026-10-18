```python
import json
import numpy as np
import math
from collections import defaultdict

def make_agent(env_spec, seed):
    with open(env_spec, 'r') as f:
        data = json.load(f)
    H = data['H']
    S = data['S']
    A = data['A']
    lambda_ = 0.60
    c_td_var = 0.01
    c_td_lin = 0.002
    c_ret_var = 0.08
    c_ret_lin = 0.01
    w_ret = 0.35
    
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Initialize Q tables
    Q_td_up = np.zeros((H+1, S, A))
    Q_td_low = np.zeros((H+1, S, A))
    Q_ret_up = np.zeros((H+1, S, A))
    Q_ret_low = np.zeros((H+1, S, A))
    
    # Initialize moments
    mean_td = np.zeros((H+1, S, A))
    second_td = np.zeros((H+1, S, A))
    mean_ret = np.zeros((H+1, S, A))
    second_ret = np.zeros((H+1, S, A))
    
    # Visit counts
    visit_counts = np.zeros((H+1, S, A), dtype=int)
    
    # Episode buffer
    episode_buffer = []
    
    def act(h, state):
        # Compute action scores
        scores = (1 - w_ret) * Q_td_up[h, state] + w_ret * Q_ret_up[h, state]
        max_score = np.max(scores)
        candidates = np.where(scores == max_score)[0]
        return rng.choice(candidates)
    
    def observe(h, state, action, reward, next_state, done):
        nonlocal episode_buffer
        episode_buffer.append((h, state, action, reward, next_state, done))
        
        if done:
            # Process episode buffer
            # Backward pass to compute returns
            G = 0.0
            for t in reversed(range(len(episode_buffer))):
                h_t, s_t, a_t, r_t, s_t1, d_t = episode_buffer[t]
                if d_t:
                    bootstrap = 0.0
                else:
                    bootstrap = np.max(Q_ret_up[h_t + 1, s_t1])
                G = r_t + lambda_ * G + (1 - lambda_) * bootstrap
                # Update return moments
                mean_ret[h_t, s_t, a_t] += (1 / (visit_counts[h_t, s_t, a_t] + 1)) * (G - mean_ret[h_t, s_t, a_t])
                second_ret[h_t, s_t, a_t] += (1 / (visit_counts[h_t, s_t, a_t] + 1)) * ((G - mean_ret[h_t, s_t, a_t]) ** 2)
                visit_counts[h_t, s_t, a_t] += 1
            
            # Compute b_ret
            L = math.log(max(10, 8 * S * A * H * 100)) + math.log(max(1, visit_counts[h, state, action]))
            var_ret = max(second_ret[h, state, action] - mean_ret[h, state, action] ** 2, 0)
            b_ret = c_ret_var * math.sqrt(var_ret * L / visit_counts[h, state, action]) + c_ret_lin * H * L / visit_counts[h, state, action]
            
            # Update Q_ret_up and Q_ret_low
            Q_ret_up[h, state, action] = np.clip(mean_ret[h, state, action] + b_ret, 0, H - h + 1)
            Q_ret_low[h, state, action] = np.clip(mean_ret[h, state, action] - b_ret, 0, H - h + 1)
            
            # Recompute state values
            for h_s in range(H + 1):
                for s in range(S):
                    Q_td_up[h_s, s] = np.max(Q_td_up[h_s, s])
                    Q_td_low[h_s, s] = np.max(Q_td_low[h_s, s])
                    Q_ret_up[h_s, s] = np.max(Q_ret_up[h_s, s])
                    Q_ret_low[h_s, s] = np.max(Q_ret_low[h_s, s])
            
            episode_buffer.clear()
    
    return act, observe
```