import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths

# Load data
calibration_lines = pd.read_csv('data/calibration_lines.csv')
calibration_transfer = pd.read_csv('data/calibration_transfer.csv')
target_lines = pd.read_csv('data/target_lines.csv')
frequency_grid = pd.read_csv('data/frequency_grid.csv')
input_pulse = pd.read_csv('data/input_pulse.csv')

# Define model parameters and loss function
def model_params(freq, R_dc, L, G, C, k_skin, calibration_fixture, normalization_gain, normalization_delay):
    Z_series = R_dc + (1 + 1j) * k_skin * np.sqrt(np.abs(freq)) + 1j * 2 * np.pi * freq * L
    Y_shunt = G + 1j * 2 * np.pi * freq * C
    gamma = np.sqrt(Z_series * Y_shunt)
    return gamma

def loss_function(params, freq, transfer, calibration_fixture, normalization_gain, normalization_delay, calibration_lines):
    R_dc, L, G, C, k_skin, calibration_fixture, normalization_gain, normalization_delay = params
    gamma = model_params(freq, R_dc, L, G, C, k_skin, calibration_fixture, normalization_gain, normalization_delay)
    line_response = np.exp(-gamma * calibration_lines['length_m'])
    predicted_transfer = transfer * line_response * normalization_gain * np.exp(1j * normalization_delay)
    return np.sum(np.abs(predicted_transfer - calibration_transfer['transfer_complex']) ** 2)

# Optimize parameters
initial_guess = [50, 0.1, 0.001, 0.01, 0.1, 1, 1, 0]
result = minimize(loss_function, initial_guess, args=(frequency_grid['frequency_hz'], calibration_transfer['transfer_complex'], calibration_lines['length_m'], calibration_lines['gain'], calibration_lines['delay']))

# Generate results
target_transfer = []
target_waveforms = []
pulse_metrics = []

for index, target in target_lines.iterrows():
    freq = frequency_grid['frequency_hz'].values
    gamma = model_params(freq, result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.x[5], result.x[6], result.x[7])
    line_response = np.exp(-gamma * target['length_m'])
    target_transfer.append({
        'target_id': target['target_id'],
        'frequency_hz': freq,
        'magnitude_db': 20 * np.log10(np.abs(line_response)),
        'phase_rad': np.angle(line_response),
        'group_delay_s': -np.gradient(np.angle(line_response), freq) / (2 * np.pi),
        'attenuation_np_per_m': -20 * np.log10(np.abs(line_response)) / target['length_m']
    })
    
    time = np.linspace(0, 1, len(input_pulse['input_v']))
    input_pulse_v = input_pulse['input_v'].values
    output_pulse_v = np.fft.irfft(np.fft.fft(input_pulse_v) * line_response)
    target_waveforms.append({
        'target_id': target['target_id'],
        'time_s': time,
        'input_v': input_pulse_v,
        'output_v': output_pulse_v
    })
    
    peaks, _ = find_peaks(output_pulse_v, height=np.median(output_pulse_v[:max(8, len(output_pulse_v)//32)]))
    fwhm, _, _ = peak_widths(output_pulse_v, peaks, rel_height=0.5)
    energy = np.trapz(output_pulse_v, time)
    tail_energy = np.sum(output_pulse_v[peaks[-1]:])
    tail_asymmetry = tail_energy / np.sum(output_pulse_v[:peaks[-1]])
    pulse_metrics.append({
        'target_id': target['target_id'],
        'input_fwhm_s': fwhm[0],
        'output_fwhm_s': fwhm[0],
        'fwhm_broadening_ratio': fwhm[0] / fwhm[0],
        'peak_delay_s': time[peaks[-1]],
        'output_peak_time_s': time[peaks[-1]],
        'energy_ratio': energy / np.trapz(input_pulse_v, time),
        'tail_asymmetry': tail_asymmetry
    })

# Save results
pd.DataFrame(target_transfer).to_csv('results/target_transfer.csv', index=False)
pd.DataFrame(target_waveforms).to_csv('results/target_waveforms.csv', index=False)
pd.DataFrame(pulse_metrics).to_csv('results/pulse_metrics.csv', index=False)