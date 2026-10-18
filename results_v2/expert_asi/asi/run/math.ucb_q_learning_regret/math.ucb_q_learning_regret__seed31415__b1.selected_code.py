import numpy as np

def make_agent(env_spec, seed):
    H, S, A, _, _ = env_spec
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    
    # Initialize Q tables
    Q_td_upper = np.zeros((H+1, S, A), dtype=np.float32)
    Q_td_lower = np.zeros((H+1, S, A), dtype=np.float32)
    Q_ret_upper = np.zeros((H+1, S, A), dtype=np.float32)
    Q_ret_lower = np.zeros((H+1, S, A), dtype=np.float32)
    
    # Moments for TD and return updates
    td_mean = np.zeros((H+1, S, A), dtype=np.float32)
    td_var = np.zeros((H+1, S, A), dtype=np.float32)
    ret_mean = np.zeros((H+1, S, A), dtype=np.float32)
    ret_var = np.zeros((H+1, S, A), dtype=np.float32)
    
    # Episode buffer
    episode = []
    
    def act(h, state):
        # Score actions using the hybrid optimistic Q estimator
        scores = (1 - 0.35) * Q_td_upper[h, state] + 0.35 * Q_ret_upper[h, state]
        return rng.choice(np.arange(A), p=np.softmax(scores))
    
    def observe(h, state, action, reward, next_state, done):
        # Append to episode buffer
        episode.append((h, state, action, reward, next_state, done))
        
        if done:
            # Process the episode buffer
            n = len(episode)
            G = 0.0
            for t in reversed(range(n)):
                h_t, s_t, a_t, r_t, s_t1, d_t = episode[t]
                if d_t:
                    bootstrap = 0.0
                else:
                    bootstrap = np.max([Q_ret_upper[h_t+1, s_t1, a] for a in range(A)])
                G = r_t + 0.60 * G + 0.40 * bootstrap
                
                # Update return moments
                weight = 1.0 / n
                ret_mean[h_t, s_t, a_t] += weight * G
                ret_var[h_t, s_t, a_t] += weight * (G**2 - ret_mean[h_t, s_t, a_t]**2)
            
            # Compute bounds for return estimates
            L = np.log(np.max([10, 8*S*A*H*1000])) + np.log(np.max([1, n]))
            b_ret = 0.08 * np.sqrt(ret_var) * np.sqrt(L / n) + 0.01 * H * L / n
            Q_ret_upper = np.clip(ret_mean + b_ret, 0, H - h)
            Q_ret_lower = np.clip(ret_mean - b_ret, 0, H - h)
            
            # Reset episode buffer
            episode.clear()
    
    return act, observe