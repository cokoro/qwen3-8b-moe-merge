import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from jax.random import PRNGKey, split, uniform, choice
from collections import defaultdict

def make_agent(env_spec, seed):
    H, S, A = env_spec['H'], env_spec['S'], env_spec['A']
    cap = H
    lambda_ = 0.60
    c_td_var, c_td_lin = 0.01, 0.002
    c_ret_var, c_ret_lin = 0.08, 0.01
    w_ret = 0.35

    rng = PRNGKey(seed)

    td_upper = jnp.zeros((H+1, S, A)) + cap
    td_lower = jnp.zeros((H+1, S, A))
    td_first = jnp.zeros((H+1, S, A))
    td_second = jnp.zeros((H+1, S, A))

    ret_upper = jnp.zeros((H+1, S, A)) + cap
    ret_lower = jnp.zeros((H+1, S, A))
    ret_first = jnp.zeros((H+1, S, A))
    ret_second = jnp.zeros((H+1, S, A))

    def act(h, state):
        nonlocal td_upper, td_lower, ret_upper, ret_lower
        # score each action by (1-w_ret)*Q_td_up[h,s,a] + w_ret*Q_ret_up[h,s,a]
        # and choose uniformly among maximizers
        scores = (1-w_ret)*(td_upper[h,state] + td_lower[h,state])/2 + w_ret*(ret_upper[h,state] + ret_lower[h,state])/2
        best_actions = jnp.argmax(scores, axis=-1)
        best_actions = jnp.where(scores == scores.max(), best_actions, -1)
        best_actions = best_actions[best_actions != -1]
        return choice(rng, best_actions, p=1/len(best_actions))

    def observe(h, state, action, reward, next_state, done):
        nonlocal td_upper, td_lower, td_first, td_second, ret_upper, ret_lower, ret_first, ret_second
        # one-step update
        n = td_first[h, state, action] + 1
        eta = (H + 1) / (H + n)
        L = np.log(max(10, 8*S*A*H*1000)) + np.log(max(1,n))
        x_up = reward + td_upper[h+1, next_state]
        x_low = reward + td_lower[h+1, next_state]
        td_first[h, state, action] += eta * x_up
        td_second[h, state, action] += eta * x_up * x_up
        td_upper[h, state, action] = min(td_upper[h, state, action], cap, td_first[h, state, action] + c_td_var*np.sqrt(td_second[h, state, action] - td_first[h, state, action]**2)*np.sqrt(L)/n + c_td_lin*cap*L/n)
        td_lower[h, state, action] = max(td_lower[h, state, action], td_first[h, state, action] - c_td_var*np.sqrt(td_second[h, state, action] - td_first[h, state, action]**2)*np.sqrt(L)/n - c_td_lin*cap*L/n, 0)

        # multi-step update
        episode.append((h, state, action, reward, next_state, done))
        if done:
            G = 0
            for h_step, s_step, a_step, r_step, ns_step, d_step in reversed(episode):
                if d_step:
                    bootstrap = 0
                else:
                    bootstrap = jnp.max(jnp.stack([ret_upper[h_step+1, ns_step, a] for a in range(A)], axis=-1))
                G = r_step + lambda_ * G + (1 - lambda_) * bootstrap
                ret_first[h_step, s_step, a_step] += 1/episode_count * G
                ret_second[h_step, s_step, a_step] += 1/episode_count * G * G
                var_ret = max(ret_second[h_step, s_step, a_step] - ret_first[h_step, s_step, a_step]**2, 0)
                b_ret = c_ret_var * np.sqrt(var_ret * L / n) + c_ret_lin * cap * L / n
                ret_upper[h_step, s_step, a_step] = min(ret_upper[h_step, s_step, a_step], cap, ret_first[h_step, s_step, a_step] + b_ret)
                ret_lower[h_step, s_step, a_step] = max(ret_lower[h_step, s_step, a_step], ret_first[h_step, s_step, a_step] - b_ret, 0)
            episode.clear()

    episode = []
    episode_count = 0
    return act, observe