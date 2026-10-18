import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.signal import find_peaks

def fit_background_period(times, values, aux_times, aux_values, n_peaks=5):
    # Find peaks in the auxiliary channels to estimate background period
    peaks, _ = find_peaks(values, height=0.5*np.max(values))
    peak_times = times[peaks]
    peak_deltas = np.diff(peak_times)
    peak_period = np.mean(peak_deltas)
    
    return peak_period

def fit_component_a_basis(times, values, background_basis, background_period):
    # Fit component A using least squares with background subtracted
    A_basis = np.zeros(2)
    def residuals(A_basis, times, values, background_basis):
        background = background_basis[0]*np.sin(2*np.pi*times/background_period) + \
                     background_basis[1]*np.cos(2*np.pi*times/background_period)
        return values - background - A_basis[0]*np.sin(2*np.pi*times) - \
               A_basis[1]*np.cos(2*np.pi*times)
    
    result = least_squares(residuals, A_basis, args=(times, values, background_basis))
    return result.x

def predict_component_a(times, A_basis, background_period):
    return A_basis[0]*np.sin(2*np.pi*times/background_period) + \
           A_basis[1]*np.cos(2*np.pi*times/background_period)

def main():
    # Read input files
    input_0 = pd.read_json('./data/input_0.json')
    input_1 = pd.read_csv('./data/input_1.csv')
    input_2 = pd.read_csv('./data/input_2.csv')
    
    # Extract times and values for all channels
    obs_0_times = input_0['time_day'].values
    obs_0_values = input_0['obs_0'].values
    obs_1_values = input_1['obs_1'].values
    obs= input_1['obs_2'].values
    obs_2_values = input_2['obs_2'].values
    
    # Estimate background period using auxiliary channels
    background_period = fit_background_period(obs_0_times, obs_0_values, obs_1_values, obs_2_values)
    
    # Fit background coefficients using auxiliary channels
    def fit_background_coefficients(times, values, background_period):
        def residuals(basis, times, values):
            return values - basis[0]*np.sin(2*np.pi*times/background_period) - \
                   basis[1]*np.cos(2*np.pi*times/background_period)
        
        result = least_squares(residuals, [0.1, 0.1], args=(times, values))
        return result.x
    
    background_basis = fit_background_coefficients(obs_0_times, obs_0_values, background_period)
    
    # Fit component A using background-subtracted values
    component_a_basis = fit_component_a_basis(obs_0_times, obs_0_values, background_basis, background_period)
    
    # Check if component A is present by comparing models
    def compare_models(background_basis, component_a_basis, obs_0_values):
        # Background-only model
        background_only = background_basis[0]*np.sin(2*np.pi*obs_0_times/background_period) + \
                          background_basis[1]*np.cos(2*np.pi*obs_0_times/background_period)
        background_error = np.mean((obs_0_values - background_only)**2)
        
        # Background-plus-A model
        background_plus_a = background_only + component_a_basis[0]*np.sin(2*np.pi*obs_0_times) + \
                           component_a_basis[1]*np.cos(2*np.pi*obs_0_times)
        background_plus_a_error = np.mean((obs_0_values - background_plus_a)**2)
        
        return background_error, background_plus_a_error
    
    background_error, background_plus_a_error = compare_models(background_basis, component_a_basis, obs_0_values)
    
    # Determine component A presence
    component_a_present = 1 if background_plus_a_error < background_error else 0
    
    # Write signal summary
    signal_summary = {
        'case_id': input_0['case_id'],
        'component_a_present': component_a_present,
        'component_a_period_day': 0.0 if component_a_present == 0 else background_period,
        'component_a_basis1_u0': 0.0 if component_a_present == 0 else component_a_basis[0],
        'component_a_basis2_u0': 0.0 if component_a_present == 0 else component_a_basis[1],
        'component_b_period_day': background_period,
        'component_b_basis1_u0': background_basis[0],
        'component_b_basis2_u0': background_basis[1],
        'component_b_basis3_u0': 0.0,
        'component_b_basis4_u0': 0.0,
        'offset_u0': 0.0
    }
    
    signal_summary_df = pd.DataFrame([signal_summary])
    signal_summary_df.to_csv('signal_summary.csv', index=False)
    
    # Predict component A values for the target times
    target_predictions = []
    for _, row in input_2.iterrows():
        time_day = row['time_day']
        component_a_value = predict_component_a(time_day, component_a_basis, background_period)
        target_predictions.append({
            'case_id': row['case_id'],
            'time_day': time_day,
            'component_a_value_u0': component_a_value if component_a_present == 1 else 0.0
        })
    
    target_predictions_df = pd.DataFrame(target_predictions)
    target_predictions_df.to_csv('target_predictions.csv', index=False)

if __name__ == '__main__':
    main()