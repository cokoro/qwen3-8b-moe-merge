```python
import numpy as np
from collections import defaultdict

def make_agent(env_spec, seed):
    H, S, A = env_spec['H'], env_spec['S'], env_spec['A']
    cap = H
    c_td_var, c_td_lin = 0.01, 0.002
    c_ret_var, c_ret_lin = 0.08, 0.01
    w_ret = 0.35
    lambda_ = 0.60

    rng = np.random.default_rng(seed)

    # One-step Q tables
    q_td_up = np.zeros((H+1, S, A), dtype=np.float64)
    q_td_low = np.zeros((H+1, S, A), dtype=np.float64)
    td_moments = [defaultdict(lambda: (0.0, 0.0)) for _ in range(H+1)]  # (mean, var)

    # Multi-step Q tables
    q_ret_up = np.zeros((H+1, S, A), dtype=np.float64)
    q_ret_low = np.zeros((H+1, S, A), dtype=np.float64)
    ret_moments = [defaultdict(lambda: (0.0, 0.0)) for _ in range(H+1)]  # (mean, var)

    # Episode buffer
    episode_buffer = []

    def act(h, state):
        # Score actions
        scores = (1 - w_ret) * q_td_up[h, state] + w_ret * q_ret_up[h, state]
        max_scores = np.max(scores)
        candidates = np.where(scores == max_scores)[0]
        return rng.choice(candidates)

    def observe(h, state, action, reward, next_state, done):
        if done:
            episode_buffer.clear()
            return

        episode_buffer.append((h, state, action, reward, next_state))

        # One-step update
        h_step, s_step, a_step = h, state, action
        n = int(td_moments[h_step][f"{s_step},{a_step}"][0] + 1)
        td_mean, td_var = td_moments[h_step][f"{s_step},{a_step}"]
        td_moments[h_step][f"{s_step},{a_step}"] = (td_mean + reward + q_td_up[h_step+1, next_state] * (1.0 / n), td_var + reward * reward + q_td_up[h_step+1, next_state] * q_td_up[h_step+1, next_state] * (1.0 / n))

        eta = (H + 1) / (H + n)
        L = np.log(max(10, 8 * S * A * H * 1000)) + np.log(max(1, n))
        var_td = max(td_moments[h_step][f"{s_step},{a_step}"][1] - (td_moments[h_step][f"{s_step},{a_step}"][0] ** 2), 0)
        b_td = c_td_var * np.sqrt(var_td * L / n) + c_td_lin * cap * L / n

        q_td_up[h_step, s_step, a_step] = min(q_td_up[h_step, s_step, a_step], cap, td_moments[h_step][f"{s_step},{a_step}"][0] + b_td)
        q_td_low[h_step, s_step, a_step] = max(q_td_low[h_step, s_step, a_step], max(0, td_moments[h_step][f"{s_step},{a_step}"][0] - b_td))

        # Recompute state values
        for s in range(S):
            q_td_up[h_step, s] = np.max(q_td_up[h_step, s])
            q_td_low[h_step, s] = np.max(q_td_low[h_step, s])

        # Multi-step update
        if done:
            G = 0.0
            for t in reversed(range(len(episode_buffer))):
                h_t, s_t, a_t, r_t, s_next_t = episode_buffer[t]
                if h_t == H:
                    bootstrap = 0.0
                else:
                    bootstrap = np.max(q_ret_up[h_t + 1, s_next_t])
                G = r_t + lambda_ * G + (1 - lambda_) * bootstrap
                ret_mean, ret_var = ret_moments[h_t][f"{s_t},{a_t}"]
                ret_moments[h_t][f"{s_t},{a_t}"] = (ret_mean + G * (1.0 / n), ret_var + G * G * (1.0 / n))

            for t in range(len(episode_buffer)):
                h_t, s_t, a_t, r_t, s_next_t = episode_buffer[t]
                ret_mean, ret_var = ret_moments[h_t][f"{s_t},{a_t}"]
                var_ret = max(ret_var - ret_mean ** 2, 0)
                b_ret = c_ret_var * np.sqrt(var_ret * L / n) + c_ret_lin * cap * L / n
                q_ret_up[h_t, s_t, a_t] = np.clip(ret_mean + b_ret, 0, cap)
                q_ret_low[h_t, s_t, a_t] = np.clip(ret_mean - b_ret, 0, cap)

            for s in range(S):
                q_ret_up[h_step, s] = np.max(q_ret_up[h_step, s])
                q_ret_low[h_step, s] = np.max(q_ret_low[h_step, s])

    return act, observe
```