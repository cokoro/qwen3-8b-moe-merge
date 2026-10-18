import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Read the data
obs_0 = pd.read_json('data/input_0.json')
obs_1 = pd.read_csv('data/input_1.csv')
obs_2 = pd.read_csv('data/input_2.csv')

# Define the model function
def model(params, t, reference_epoch_day, P_A, P_B):
    offset, A1, A2, B1, B2, B3, B4 = params
    t = t - reference_epoch_day
    return offset + A1 * np.sin(2 * np.pi * t / P_A) + A2 * np.cos(2 * np.pi * t / P_A) + B1 * np.sin(2 * np.pi * t / P_B) + B2 * np.cos(2 * np.pi * t / P_B) + B3 * np.sin(4 * np.pi * t / P_B) + B4 * np.cos(4 * np.pi * t / P_B)

# Define the cost function for optimization
def cost(params, t, y, reference_epoch_day, P_A, P_B):
    return np.sum((model(params, t, reference_epoch_day, P_A, P_B) - y) ** 2)

# Estimate background period from auxiliary channels
def estimate_background_period(obs_1, obs_2):
    # Use FFT on the auxiliary channels to estimate the background period
    freq_1 = np.fft.fft(obs_1['value']) / np.fft.fft(obs_1['value']).max()
    freq_2 = np.fft.fft(obs_2['value']) / np.fft.fft(obs_2['value']).max()
    peak_freq_1 = np.argmax(freq_1) / len(freq_1)
    peak_freq_2 = np.argmax(freq_2) / len(freq_2)
    return 1 / peak_freq_1, 1 / peak_freq_2

# Fit the model for each case
cases = obs_0['case_id'].unique()
results = []

for case_id in cases:
    obs_case = obs_0[obs_0['case_id'] == case_id]
    reference_epoch_day = obs_case['reference_epoch_day'].values[0]
    obs_1_case = obs_1[obs_1['case_id'] == case_id]
    obs_2_case = obs_2[obs_2['case_id'] == case_id]

    # Estimate background period from auxiliary channels
    P_B_1, P_B_2 = estimate_background_period(obs_1_case, obs_2_case)
    P_B = (P_B_1 + P_B_2) / 2

    # Refine background period using all auxiliary channels
    t_all = np.concatenate([obs_1_case['time_day'].values, obs_2_case['time_day'].values])
    y_all = np.concatenate([obs_1_case['value'].values, obs_2_case['value'].values])
    params_initial = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    result = minimize(lambda x: cost(x, t_all, y_all, reference_epoch_day, 0.0, P_B), params_initial, tol=1e-6)
    B1, B2, B3, B4 = result.x[2], result.x[3], result.x[5], result.x[6]

    # Refine A period using the background estimate
    P_A_candidates = np.array([24.96, 12.48, 10.66])
    P_A = P_A_candidates[0]
    for _ in range(10):
        params_initial = np.array([0.0, 1.0, 0.0, B1, B2, B3, B4])
        result = minimize(lambda x: cost(x, obs_case['time_day'].values, obs_case['value'].values, reference_epoch_day, P_A, P_B), params_initial, tol=1e-6)
        offset, A1, A2, B1, B2, B3, B4 = result.x
        P_A = 24.96 / (B1 / np.sin(2 * np.pi * P_A / P_B) + B2 / np.cos(2 * np.pi * P_A / P_B)) * P_B

    # Refine A period and coefficients using leave-one-out cross-validation
    t_loo = obs_case['time_day'].values
    y_loo = obs_case['value'].values
    for _ in range(10):
        params_initial = np.array([0.0, 1.0, 0.0, B1, B2, B3, B4])
        result = minimize(lambda x: cost(x, t_loo, y_loo, reference_epoch_day, P_A, P_B), params_initial, tol=1e-6)
        offset, A1, A2, B1, B2, B3, B4 = result.x

    # Check for no-A case
    if np.allclose(B1, 0) and np.allclose(B2, 0) and np.allclose(B3, 0) and np.allclose(B4, 0):
        component_a_present = 0
    else:
        component_a_present = 1

    results.append({
        'case_id': case_id,
        'component_a_present': component_a_present,
        'component_a_period_day': P_A,
        'component_a_basis1_u0': A1,
        'component_a_basis2_u0': A2,
        'component_b_period_day': P_B,
        'component_b_basis1_u0': B1,
        'component_b_basis2_u0': B2,
        'component_b_basis3_u0': B3,
        'component_b_basis4_u0': B4,
        'offset_u0': offset
    })

# Write the signal summary
pd.DataFrame(results).to_csv('signal_summary.csv', index=False)

# Predict the future component-A-only values
obs_2['component_a_value_u0'] = 0.0
for case_id in cases:
    obs_2_case = obs_2[obs_2['case_id'] == case_id]
    obs_2_case['component_a_value_u0'] = results[obs_2_case['case_id'].values[0]]['component_a_basis1_u0'] * np.sin(2 * np.pi * (obs_2_case['time_day'].values - reference_epoch_day) / results[obs_2_case['case_id'].values[0]]['component_a_period_day']) + results[obs_2_case['case_id'].values[0]]['component_a_basis2_u0'] * np.cos(2 * np.pi * (obs_2_case['time_day'].values - reference_epoch_day) / results[obs_2_case['case_id'].values[0]]['component_a_period_day'])
    obs_2.loc[obs_2_case.index, 'component_a_value_u0'] = obs_2_case['component_a_value_u0'].values

obs_2.to_csv('target_predictions.csv', index=False)