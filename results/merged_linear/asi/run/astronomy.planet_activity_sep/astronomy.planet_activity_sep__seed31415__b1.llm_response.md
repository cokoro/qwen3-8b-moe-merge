```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import json

def load_data():
    with open('./data/input_0.json') as f:
        case_data = json.load(f)
    obs1 = pd.read_csv('./data/input_1.csv')
    obs2 = pd.read_csv('./data/input_2.csv')
    return case_data, obs1, obs2

def fit_background_period(obs1, obs2):
    t1 = obs1['time_day'].values
    y1 = obs1['observed'].values
    t2 = obs2['time_day'].values
    y2 = obs2['observed'].values

    def residuals(params, t, y):
        P = params[0]
        y_pred = params[1] * np.sin(2*np.pi*t/P) + params[2] * np.cos(2*np.pi*t/P)
        return y - y_pred

    def fit_period(t1, y1, t2, y2):
        res = least_squares(lambda x: np.concatenate([residuals(x, t1, y1), residuals(x, t2, y2)]), x0=[28, 0, 0], bounds=(None, [100, 1, 1]))
        return res.x[0]

    return fit_period(t1, y1, t2, y2)

def fit_component_a(obs0, P_a, P_b):
    t0 = obs0['time_day'].values
    y0 = obs0['observed'].values
    t_ref = obs0['reference_epoch_day'].values[0]
    t = t0 - t_ref

    def model(params, t, P_a, P_b):
        offset = params[0]
        A1 = params[1]
        A2 = params[2]
        B1 = params[3]
        B2 = params[4]
        B3 = params[5]
        B4 = params[6]
        return offset + A1*np.sin(2*np.pi*t/P_a) + A2*np.cos(2*np.pi*t/P_a) + B1*np.sin(2*np.pi*t/P_b) + B2*np.cos(2*np.pi*t/P_b) + B3*np.sin(4*np.pi*t/P_b) + B4*np.cos(4*np.pi*t/P_b)

    def residuals(params):
        return model(params, t, P_a, P_b) - y0

    x0 = [0, 0, 0, 0, 0, 0, 0]
    result = least_squares(residuals, x0)
    return result.x

def compute_forecasts(P_a, A1, A2, t_pred):
    t_ref = t_pred[0] - t_pred[1]
    return A1*np.sin(2*np.pi*(t_pred - t_ref)/P_a) + A2*np.cos(2*np.pi*(t_pred - t_ref)/P_a)

def main():
    case_data, obs1, obs2 = load_data()
    P_b = fit_background_period(obs1, obs2)
    results = []

    for case_id in case_data:
        obs0 = pd.DataFrame({
            'time_day': case_data[case_id]['observed']['time_day'],
            'observed': case_data[case_id]['observed']['observed'],
            'reference_epoch_day': case_data[case_id]['observed']['reference_epoch_day']
        })
        t_ref = obs0['reference_epoch_day'].values[0]
        t = obs0['time_day'].values - t_ref
        y = obs0['observed'].values

        def fit_period_candidate(P_a, P_b):
            params = fit_component_a(obs0, P_a, P_b)
            offset = params[0]
            A1 = params[1]
            A2 = params[2]
            B1 = params[3]
            B2 = params[4]
            B3 = params[5]
            B4 = params[6]
            return {
                'component_a_present': 1,
                'component_a_period_day': P_a,
                'component_a_basis1_u0': A1,
                'component_a_basis2_u0': A2,
                'component_b_period_day': P_b,
                'component_b_basis1_u0': B1,
                'component_b_basis2_u0': B2,
                'component_b_basis3_u0': B3,
                'component_b_basis4_u0': B4,
                'offset_u0': offset
            }

        def fit_no_component_a():
            params = fit_component_a(obs0, 0, P_b)
            offset = params[0]
            B1 = params[3]
            B2 = params[4]
            B3 = params[5]
            B4 = params[6]
            return {
                'component_a_present': 0,
                'component_a_period_day': 0,
                'component_a_basis1_u0': 0.0,
                'component_a_basis2_u0': 0.0,
                'component_b_period_day': P_b,
                'component_b_basis1_u0': B1,
                'component_b_basis2_u0': B2,
                'component_b_basis3_u0': B3,
                'component_b_basis4_u0': B4,
                'offset_u0': offset
            }

        def compare_models(candidate, no_component_candidate):
            candidate_resid = np.sum((y - candidate['offset_u0'] - candidate['component_a_basis1_u0']*np.sin(2*np.pi*t/candidate['component_a_period_day']) - candidate['component_a_basis2_u0']*np.cos(2*np.pi*t/candidate['component_a_period_day']) - candidate['component_b_basis1_u0']*np.sin(2*np.pi*t/candidate['component_b_period_day']) - candidate['component_b_basis2_u0']*np.cos(2*np.pi*t/candidate['component_b_period_day']) - candidate['component_b_basis3_u0']*np.sin(4*np.pi*t/candidate['component_b_period_day']) - candidate['component_b_basis4_u0']*np.cos(4*np.pi*t/candidate['component_b_period_day']))**2)
            no_component_resid = np.sum((y - no_component_candidate['offset_u0'] - no_component_candidate['component_b_basis1_u0']*np.sin(2*np.pi*t/no_component_candidate['component_b_period_day']) - no_component_candidate['component_b_basis2_u0']*np.cos(2*np.pi*t/no_component_candidate['component_b_period_day']) - no_component_candidate['component_b_basis3_u0']*np.sin(4*np.pi*t/no_component_candidate['component_b_period_day']) - no_component_candidate['component_b_basis4_u0']*np.cos(4*np.pi*t/no_component_candidate['component_b_period_day']))**2)
            return candidate_resid < no_component_resid

        if compare_models(fit_period_candidate(24.96, P_b), fit_no_component_a()):
            results.append(fit_period_candidate(24.96, P_b))
        else:
            results.append(fit_no_component_a())

    signal_summary = pd.DataFrame(results)
    signal_summary.to_csv('signal_summary.csv', index=False)

    target_predictions = pd.read_csv('./data/input_2.csv')
    target_predictions['case_id'] = target_predictions['case_id'].astype(str)
    target_predictions['component_a_value_u0'] = 0.0
    for _, row in target_predictions.iterrows():
        case_id = row['case_id']
        time_day = row['time_day']
        for idx, result in enumerate(results):
            if result['case_id'] == case_id:
                P_a = result['component_a_period_day']
                A1 = result['component_a_basis1_u0']
                A2 = result['component_a_basis2_u0']
                t_pred = [time_day, P_a]
                forecast = compute_forecasts(P_a, A1, A2, t_pred)
                target_predictions.at[idx, 'component_a_value_u0'] = forecast[0]
                break

    target_predictions.to_csv('target_predictions.csv', index=False)

if __name__ == '__main__':
    main()
```