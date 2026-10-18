import numpy as np
import pandas as pd
import json
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load input data
with open('./data/input_0.json', 'r') as f:
    input_0 = json.load(f)

input_1 = pd.read_csv('./data/input_1.csv')
input_2 = pd.read_csv('./data/input_2.csv')

# Define reference epoch
reference_epoch_day = input_0['reference_epoch_day']

# Function to compute time offset
def time_offset(t):
    return t - reference_epoch_day

# Function to compute sine and cosine terms
def compute_basis(t, period):
    t_norm = t / period
    return np.sin(2 * np.pi * t_norm), np.cos(2 * np.pi * t_norm)

# Function to compute harmonic terms
def compute_harmonic_basis(t, period):
    t_norm = t / period
    return np.sin(4 * np.pi * t_norm), np.cos(4 * np.pi * t_norm)

# Function to fit model for a given P_B
def fit_model(t, obs_0, P_B, P_A=None, component_a_present=True):
    if P_A is None:
        P_A = P_B / 2  # Assume P_A is half of P_B for initial estimate

    # Prepare basis terms
    t_offset = time_offset(t)
    basis1_0, basis2_0 = compute_basis(t_offset, P_A)
    basis1_b, basis2_b = compute_basis(t_offset, P_B)
    basis3_b, basis4_b = compute_harmonic_basis(t_offset, P_B)

    # Combine all basis terms
    X = np.column_stack([
        basis1_0, basis2_0, basis1_b, basis2_b, basis3_b, basis4_b
    ])

    # Fit linear model
    model = LinearRegression().fit(X, obs_0)
    coefficients = model.coef_
    offset = model.intercept_

    # Extract coefficients
    A1, A2, B1, B2, B3, B4 = coefficients
    offset_u0 = offset

    # If component A is not present, set A coefficients to zero
    if not component_a_present:
        A1, A2 = 0.0, 0.0

    return A1, A2, B1, B2, B3, B4, offset_u0

# Function to estimate P_B from auxiliary channels
def estimate_P_B(t, obs_1, obs_2):
    # Compute time offset
    t_offset = time_offset(t)

    # Compute sine and cosine terms for possible P_B
    # Use peaks to estimate period
    peaks_1, _ = find_peaks(obs_1)
    peaks_2, _ = find_peaks(obs_2)

    # Estimate P_B from peaks
    P_B_1 = np.mean(np.diff(t[peaks_1]))
    P_B_2 = np.mean(np.diff(t[peaks_2]))

    # Average the estimates
    P_B = (P_B_1 + P_B_2) / 2

    return P_B

# Function to compute component A values for future times
def predict_component_A(t, P_A, A1, A2):
    t_offset = time_offset(t)
    basis1, basis2 = compute_basis(t_offset, P_A)
    return A1 * basis1 + A2 * basis2

# Estimate P_B from auxiliary channels
t = np.concatenate([input_1['time_day'], input_2['time_day']])
obs_1_values = input_1['obs_1'].values
obs_2_values = input_2['obs_2'].values

P_B = estimate_P_B(t, obs_1_values, obs_2_values)

# Fit model for obs_0
obs_0_values = input_0['obs_0'].values
t_obs_0 = input_0['time_day'].values
t_obs_0_offset = time_offset(t_obs_0)

# Fit model with P_B and P_A as half of P_B
A1, A2, B1, B2, B3, B4, offset_u0 = fit_model(t_obs_0_offset, obs_0_values, P_B, P_B / 2)

# Check if component A is present
# Compare background-only model with background-plus-A model
# For simplicity, use a heuristic based on residual
obs_0_background = B1 * np.sin(2 * np.pi * t_obs_0_offset / P_B) + B2 * np.cos(2 * np.pi * t_obs_0_offset / P_B) + B3 * np.sin(4 * np.pi * t_obs_0_offset / P_B) + B4 * np.cos(4 * np.pi * t_obs_0_offset / P_B) + offset_u0
residual_background = np.mean((obs_0_values - obs_0_background) ** 2)

obs_0_full = A1 * np.sin(2 * np.pi * t_obs_0_offset / P_B / 2) + A2 * np.cos(2 * np.pi * t_obs_0_offset / P_B / 2) + B1 * np.sin(2 * np.pi * t_obs_0_offset / P_B) + B2 * np.cos(2 * np.pi * t_obs_0_offset / P_B) + B3 * np.sin(4 * np.pi * t_obs_0_offset / P_B) + B4 * np.cos(4 * np.pi * t_obs_0_offset / P_B) + offset_u0
residual_full = np.mean((obs_0_values - obs_0_full) ** 2)

component_a_present = 1 if residual_full < residual_background else 0

# Write signal_summary.csv
signal_summary = pd.DataFrame({
    'case_id': [input_0['case_id']],
    'component_a_present': [component_a_present],
    'component_a_period_day': [P_B / 2 if component_a_present else 0.0],
    'component_a_basis1_u0': [A1],
    'component_a_basis2_u0': [A2],
    'component_b_period_day': [P_B],
    'component_b_basis1_u0': [B1],
    'component_b_basis2_u0': [B2],
    'component_b_basis3_u0': [B3],
    'component_b_basis4_u0': [B4],
    'offset_u0': [offset_u0]
})

signal_summary.to_csv('signal_summary.csv', index=False)

# Predict component A values for input_2
input_2['component_a_value_u0'] = input_2['time_day'].apply(lambda t: predict_component_A(t, P_B / 2 if component_a_present else None, A1, A2))

# Write target_predictions.csv
input_2[['case_id', 'time_day', 'component_a_value_u0']].to_csv('target_predictions.csv', index=False)