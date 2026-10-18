import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import minimize
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load input data
input_0_path = './data/input_0.json'
input_1_path = './data/input_1.csv'
input_2_path = './data/input_2.csv'

with open(input_0_path, 'r') as f:
    input_0 = json.load(f)

input_1 = pd.read_csv(input_1_path)
input_2 = pd.read_csv(input_2_path)

# Extract reference epoch and case IDs
reference_epoch_day = input_0['reference_epoch_day']
case_ids = input_0['case_ids']

# Function to extract time and observed values
def extract_data(df, time_col='time_day', obs_col='obs'):
    return df[[time_col, obs_col]].values

# Extract data from each channel
obs_0_data = extract_data(input_0['obs_0'])
obs_1_data = extract_data(input_1)
obs_2_data = extract_data(input_2)

# Function to estimate background period from auxiliary channels
def estimate_background_period(obs_1, obs_2):
    # Convert to numpy arrays
    t1, y1 = obs_1.T
    t2, y2 = obs_2.T

    # Compute time differences
    dt = t2 - t1

    # Find peaks in both channels
    peaks1, _ = find_peaks(y1, height=np.mean(y1))
    peaks2, _ = find_peaks(y2, height=np.mean(y2))

    # Estimate period from time differences between peaks
    period_candidates = []
    for p1 in peaks1:
        for p2 in peaks2:
            if abs(t2[p2] - t1[p1]) > 0:
                period_candidates.append(abs(t2[p2] - t1[p1]))

    # Use mode of period candidates
    period = np.round(np.mean(period_candidates), 2)
    return period

# Estimate background period from auxiliary channels
background_period = estimate_background_period(obs_1_data, obs_2_data)

# Function to fit component A and B for a given P_A and P_B
def fit_model(obs_0, P_A, P_B):
    t, y = obs_0.T
    t_shifted = t - reference_epoch_day

    # Define model
    def model(params, t):
        offset, A1, A2, B1, B2, B3, B4 = params
        return (offset + 
                A1 * np.sin(2 * np.pi * t / P_A) + 
                A2 * np.cos(2 * np.pi * t / P_A) + 
                B1 * np.sin(2 * np.pi * t / P_B) + 
                B2 * np.cos(2 * np.pi * t / P_B) + 
                B3 * np.sin(4 * np.pi * t / P_B) + 
                B4 * np.cos(4 * np.pi * t / P_B))

    # Initial guess
    initial_guess = [np.mean(y), 0, 0, 0, 0, 0, 0]

    # Fit model
    result = minimize(lambda params: np.sum((model(params, t_shifted) - y) ** 2), initial_guess)
    params = result.x

    # Extract coefficients
    offset, A1, A2, B1, B2, B3, B4 = params
    return offset, A1, A2, B1, B2, B3, B4

# Function to evaluate model fit
def evaluate_fit(obs_0, P_A, P_B, model_params):
    t, y = obs_0.T
    t_shifted = t - reference_epoch_day
    y_pred = model_params[0] + model_params[1] * np.sin(2 * np.pi * t_shifted / P_A) + \
             model_params[2] * np.cos(2 * np.pi * t_shifted / P_A) + \
             model_params[3] * np.sin(2 * np.pi * t_shifted / P_B) + \
             model_params[4] * np.cos(2 * np.pi * t_shifted / P_B) + \
             model_params[5] * np.sin(4 * np.pi * t_shifted / P_B) + \
             model_params[6] * np.cos(4 * np.pi * t_shifted / P_B)
    return mean_squared_error(y, y_pred)

# Function to compare models
def compare_models(obs_0, P_B, candidates):
    model_params = {}
    for P_A in candidates:
        params = fit_model(obs_0, P_A, P_B)
        model_params[P_A] = params
    return model_params

# Function to determine component A presence
def determine_component_a_presence(model_params, obs_0, P_A, P_B):
    # Background-only model
    params_bg = fit_model(obs_0, 0, P_B)
    mse_bg = evaluate_fit(obs_0, 0, P_B, params_bg)

    # Background-plus-A model
    params_bg_a = model_params[P_A]
    mse_bg_a = evaluate_fit(obs_0, P_A, P_B, params_bg_a)

    # Check if A improves the fit
    if mse_bg_a < mse_bg:
        return 1, params_bg_a
    else:
        return 0, params_bg

# Function to predict future component A values
def predict_component_a_future(params, P_A, times):
    t_shifted = times - reference_epoch_day
    return params[1] * np.sin(2 * np.pi * t_shifted / P_A) + params[2] * np.cos(2 * np.pi * t_shifted / P_A)

# Generate candidate periods for component A
candidate_periods = {
    'case_000': np.linspace(24.82, 25.08, 100),
    'case_001': np.linspace(12.36, 12.60, 100),
    'case_002': np.linspace(28.0, 28.0, 1),
    'case_003': np.linspace(10.54, 10.72, 100)
}

# Process each case
signal_summary = []
target_predictions = []

for case_id in case_ids:
    # Extract observed data for this case
    obs_0_case = obs_0_data[obs_0_data[:, 0] == case_id, :]
    obs_0_case = obs_0_case[obs_0_case[:, 0].astype(int) == case_id, :]

    # Fit background model
    model_params_bg = fit_model(obs_0_case, 0, background_period)

    # Compare models for each candidate A period
    best_model = None
    best_mse = float('inf')
    best_P_A = None

    for case_key, P_A_candidates in candidate_periods.items():
        for P_A in P_A_candidates:
            model_params = fit_model(obs_0_case, P_A, background_period)
            mse = evaluate_fit(obs_0_case, P_A, background_period, model_params)
            if mse < best_mse:
                best_mse = mse
                best_model = model_params
                best_P_A = P_A

    # Determine presence of component A
    component_a_present, model_params = determine_component_a_presence(best_model, obs_0_case, best_P_A, background_period)

    # Extract coefficients
    offset, A1, A2, B1, B2, B3, B4 = model_params
    component_a_period_day = best_P_A if component_a_present else 0.0
    component_a_basis1_u0 = A1 if component_a_present else 0.0
    component_a_basis2_u0 = A2 if component_a_present else 0.0
    component_b_period_day = background_period
    component_b_basis1_u0 = B1
    component_b_basis2_u0 = B2
    component_b_basis3_u0 = B3
    component_b_basis4_u0 = B4
    offset_u0 = offset

    # Append to signal summary
    signal_summary.append([
        case_id,
        component_a_present,
        component_a_period_day,
        component_a_basis1_u0,
        component_a_basis2_u0,
        component_b_period_day,
        component_b_basis1_u0,
        component_b_basis2_u0,
        component_b_basis3_u0,
        component_b_basis4_u0,
        offset_u0
    ])

    # Predict future component A values
    if component_a_present:
        future_times = input_2[input_2['case_id'] == case_id]['time_day'].values
        future_values = predict_component_a_future(model_params, best_P_A, future_times)
        for time, value in zip(future_times, future_values):
            target_predictions.append([
                case_id,
                time,
                value
            ])
    else:
        # No component A, set values to zero
        future_times = input_2[input_2['case_id'] == case_id]['time_day'].values
        for time in future_times:
            target_predictions.append([
                case_id,
                time,
                0.0
            ])

# Save results
signal_df = pd.DataFrame(signal_summary, columns=[
    'case_id', 'component_a_present', 'component_a_period_day', 'component_a_basis1_u0',
    'component_a_basis2_u0', 'component_b_period_day', 'component_b_basis1_u0', 'component_b_basis2_u0',
    'component_b_basis3_u0', 'component_b_basis4_u0', 'offset_u0'
])
signal_df.to_csv('signal_summary.csv', index=False)

target_df = pd.DataFrame(target_predictions, columns=['case_id', 'time_day', 'component_a_value_u0'])
target_df.to_csv('target_predictions.csv', index=False)