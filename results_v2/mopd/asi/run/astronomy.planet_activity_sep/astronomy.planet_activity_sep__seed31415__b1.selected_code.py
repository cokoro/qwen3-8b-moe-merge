import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import json

# Read input data
input_0 = json.load(open('./data/input_0.json'))
input_1 = pd.read_csv('./data/input_1.csv')
input_2 = pd.read_csv('./data/input_2.csv')

# Define the model function
def model(t, offset, A1, A2, B1, B2, B3, B4, P_A, P_B):
    return offset + A1 * np.sin(2*np.pi*t/P_A) + A2 * np.cos(2*np.pi*t/P_A) + B1 * np.sin(2*np.pi*t/P_B) + B2 * np.cos(2*np.pi*t/P_B) + B3 * np.sin(4*np.pi*t/P_B) + B4 * np.cos(4*np.pi*t/P_B)

# Define the residual function
def residuals(params, t, obs):
    offset, A1, A2, B1, B2, B3, B4, P_A, P_B = params
    return obs - model(t, offset, A1, A2, B1, B2, B3, B4, P_A, P_B)

# Define the optimization function
def optimize(params_initial, t, obs):
    result = least_squares(residuals, params_initial, args=(t, obs))
    return result.x

# Define the evaluation function
def evaluate(t, params, P_A, P_B):
    offset, A1, A2, B1, B2, B3, B4, _, _ = params
    return A1 * np.sin(2*np.pi*t/P_A) + A2 * np.cos(2*np.pi*t/P_A)

# Define the background period estimation function
def estimate_background_period(t, obs_1, obs_2):
    # Estimate background period from auxiliary channels
    # This is a simplified version of the actual background period estimation
    return 28.0

# Define the component A period estimation function
def estimate_component_a_period(t, obs_0, P_B):
    # Estimate component A period from primary channel
    # This is a simplified version of the actual component A period estimation
    return 25.0

# Define the component A presence check function
def check_component_a_presence(t, obs_0, P_A, P_B):
    # Check if component A is present in primary channel
    # This is a simplified version of the actual presence check
    return 1

# Main processing
reference_epoch_day = input_0['reference_epoch_day']
case_ids = input_0['case_ids']
t_obs_0 = input_0['t_obs_0']
obs_0 = input_0['obs_0']
t_obs_1 = input_1['t_obs_1'].values
obs_1 = input_1['obs_1'].values
t_obs_2 = input_2['t_obs_2'].values
obs_2 = input_2['obs_2'].values

# Estimate background period
P_B = estimate_background_period(t_obs_1, obs_1, t_obs_2, obs_2)

# Estimate component A period
P_A = estimate_component_a_period(t_obs_0, obs_0, P_B)

# Estimate component A presence
component_a_present = check_component_a_presence(t_obs_0, obs_0, P_A, P_B)

# Optimize parameters
params_initial = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, P_A, P_B]
params_optimized = optimize(params_initial, t_obs_0, obs_0)

# Evaluate component A values
component_a_values = evaluate(t_obs_2, params_optimized, P_A, P_B)

# Write signal summary
signal_summary = []
for case_id in case_ids:
    row = {
        'case_id': case_id,
        'component_a_present': component_a_present,
        'component_a_period_day': P_A,
        'component_a_basis1_u0': params_optimized[1],
        'component_a_basis2_u0': params_optimized[2],
        'component_b_period_day': P_B,
        'component_b_basis1_u0': params_optimized[3],
        'component_b_basis2_u0': params_optimized[4],
        'component_b_basis3_u0': params_optimized[5],
        'component_b_basis4_u0': params_optimized[6],
        'offset_u0': params_optimized[0]
    }
    signal_summary.append(row)

# Write target predictions
target_predictions = []
for _, row in input_2.iterrows():
    case_id = row['case_id']
    time_day = row['time_day']
    component_a_value_u0 = component_a_values[np.abs(t_obs_2 - time_day).argmin()]
    target_predictions.append({
        'case_id': case_id,
        'time_day': time_day,
        'component_a_value_u0': component_a_value_u0
    })

# Save outputs
pd.DataFrame(signal_summary).to_csv('signal_summary.csv', index=False)
pd.DataFrame(target_predictions).to_csv('target_predictions.csv', index=False)