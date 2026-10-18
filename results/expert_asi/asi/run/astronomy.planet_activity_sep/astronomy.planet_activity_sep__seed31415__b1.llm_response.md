```python
import numpy as np
import pandas as pd
import json

def load_data(path):
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    else:
        return pd.read_csv(path)

def fit_period(periods, t, y, n_periods=5):
    def cost(p):
        w = np.exp(-0.1 * (p - periods)**2 / (10**2))
        return np.sqrt(np.sum(w * (y - np.sin(2*np.pi*p*t)**2)**2))
    return minimize(cost, periods[0], bounds=(periods[0]-0.1, periods[1]+0.1)).x

def main():
    obs_0 = load_data('./data/input_0.json')
    obs_1 = load_data('./data/input_1.csv')
    obs_2 = load_data('./data/input_2.csv')

    # Estimate background period
    t1 = obs_1['time_day'].values
    y1 = obs_1['value'].values
    t2 = obs_2['time_day'].values
    y2 = obs_2['value'].values

    p_b = fit_period([24, 28], t1, y1, n_periods=5)
    p_b = fit_period([24, 28], t2, y2, n_periods=5)

    # Fit component A
    t = obs_0['time_day'].values
    y = obs_0['value'].values
    t_ref = obs_0['reference_epoch_day'][0]

    # Try multiple candidate A periods
    candidates = []
    for p_a in [24.96, 12.36, 10.54]:
        t_ = t - t_ref
        A = np.vstack([np.sin(2*np.pi*p_a*t_), np.cos(2*np.pi*p_a*t_)])
        B = np.vstack([np.sin(2*np.pi*p_b*t_), np.cos(2*np.pi*p_b*t_)])
        B = np.hstack([B, np.sin(4*np.pi*p_b*t_), np.cos(4*np.pi*p_b*t_)])
        X = np.hstack([A, B])
        coefs = np.linalg.lstsq(X, y, rcond=None)[0]
        candidates.append((p_a, coefs))

    # Select best candidate
    best = min(candidates, key=lambda x: np.sum((x[1][2:] - np.mean(y))**2))

    # Write summary
    summary = {
        'case_id': 'case_000',
        'component_a_present': 1,
        'component_a_period_day': best[0],
        'component_a_basis1_u0': best[1][0],
        'component_a_basis2_u0': best[1][1],
        'component_b_period_day': p_b,
        'component_b_basis1_u0': best[1][2],
        'component_b_basis2_u0': best[1][3],
        'component_b_basis3_u0': best[1][4],
        'component_b_basis4_u0': best[1][5],
        'offset_u0': best[1][6],
    }

    # Predict future
    future = []
    for _, row in obs_2.iterrows():
        t_ = row['time_day'] - t_ref
        a = summary['component_a_basis1_u0'] * np.sin(2*np.pi*summary['component_a_period_day']*t_) + \
            summary['component_a_basis2_u0'] * np.cos(2*np.pi*summary['component_a_period_day']*t_)
        future.append({
            'case_id': row['case_id'],
            'time_day': row['time_day'],
            'component_a_value_u0': a,
        })

    pd.DataFrame(summary).to_csv('signal_summary.csv', index=False)
    pd.DataFrame(future).to_csv('target_predictions.csv', index=False)

if __name__ == '__main__':
    main()
```