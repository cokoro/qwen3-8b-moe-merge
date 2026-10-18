```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_widths
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load data
calibration_lines = pd.read_csv('data/calibration_lines.csv')
calibration_transfer = pd.read_csv('data/calibration_transfer.csv')
target_lines = pd.read_csv('data/target_lines.csv')
frequency_grid = pd.read_csv('data/frequency_grid.csv')
input_pulse = pd.read_csv('data/input_pulse.csv')

# Extract frequency grid
frequency_hz = frequency_grid['frequency_hz'].values
num_freq = len(frequency_hz)

# Extract input pulse
time_s = input_pulse['time_s'].values
input_v = input_pulse['input_v'].values

# Define model parameters
def model_params():
    return {
        'R_dc': 50.0,  # Ohms
        'L': 0.1,      # H/m
        'G': 1e-6,     # S/m
        'C': 1e-12,    # F/m
        'k_skin': 1.0,  # Skin depth factor
        'shared_fixture': np.ones(num_freq) + 1j * np.zeros(num_freq),
        'line_gain': np.ones(num_freq) + 1j * np.zeros(num_freq),
        'line_delay': np.zeros(num_freq)
    }

# Define the intrinsic line transfer function
def H_line(f, length_m, width_m, roughness, params):
    omega = 2 * np.pi * f
    Z_series = params['R_dc'] + (1 + 1j) * params['k_skin'] * np.sqrt(np.abs(f)) + 1j * omega * params['L']
    Y_shunt = params['G'] + 1j * omega * params['C']
    gamma = np.sqrt(Z_series * Y_shunt)
    H = np.exp(-gamma * length_m)
    return H

# Define the calibration model
def calibration_model(params, calibration_lines, calibration_transfer):
    # Shared fixture and line normalization
    shared_fixture = params['shared_fixture']
    line_gain = params['line_gain']
    line_delay = params['line_delay']
    
    # Intrinsic line transfer functions
    H_line_all = np.zeros((len(calibration_lines), num_freq), dtype=complex)
    for i, row in calibration_lines.iterrows():
        length_m = row['length_m']
        width_m = row['width_m']
        roughness = row['roughness']
        H_line_all[i] = H_line(frequency_hz, length_m, width_m, roughness, params)
    
    # Predicted transfer functions
    H_pred = shared_fixture * H_line_all * line_gain * np.exp(1j * omega * line_delay)
    
    # Residuals
    H_measured = calibration_transfer.values
    residuals = np.abs(H_measured - H_pred)
    
    return np.sum(residuals ** 2)

# Optimize parameters
initial_params = model_params()
result = minimize(calibration_model, initial_params, args=(calibration_lines, calibration_transfer), method='L-BFGS-B')
optimized_params = result.x

# Compute intrinsic line transfer functions for target lines
target_transfer = np.zeros((len(target_lines), num_freq), dtype=complex)
for i, row in target_lines.iterrows():
    length_m = row['length_m']
    width_m = row['width_m']
    roughness = row['roughness']
    target_transfer[i] = H_line(frequency_hz, length_m, width_m, roughness, optimized_params)

# Convert to dB and group delay
magnitude_db = 20 * np.log10(np.abs(target_transfer))
phase_rad = np.unwrap(np.angle(target_transfer))
group_delay_s = -np.gradient(phase_rad, frequency_hz) / (2 * np.pi)
attenuation_np_per_m = 20 * np.log10(np.abs(target_transfer) / np.abs(target_transfer[0]))

# Save target transfer results
target_transfer_df = pd.DataFrame({
    'target_id': target_lines['target_id'],
    'frequency_hz': frequency_hz,
    'magnitude_db': magnitude_db,
    'phase_rad': phase_rad,
    'group_delay_s': group_delay_s,
    'attenuation_np_per_m': attenuation_np_per_m
})
target_transfer_df.to_csv('results/target_transfer.csv', index=False)

# Generate waveforms
waveforms = []
for i, row in target_lines.iterrows():
    length_m = row['length_m']
    width_m = row['width_m']
    roughness = row['roughness']
    H = H_line(frequency_hz, length_m, width_m, roughness, optimized_params)
    H_fft = np.fft.fftshift(np.fft.fft(H))
    input_fft = np.fft.fftshift(np.fft.fft(input_v))
    output_fft = H_fft * input_fft
    output_v = np.fft.ifft(np.fft.ifftshift(output_fft)).real
    waveforms.append({
        'target_id': row['target_id'],
        'time_s': time_s,
        'input_v': input_v,
        'output_v': output_v
    })
waveforms_df = pd.DataFrame(waveforms)
waveforms_df.to_csv('results/target_waveforms.csv', index=False)

# Compute pulse metrics
pulse_metrics = []
for i, row in target_lines.iterrows():
    output_v = waveforms[i]['output_v']
    peaks, _ = find_peaks(output_v, height=np.median(output_v[:max(8, len(output_v)//32)]))
    if len(peaks) == 0:
        continue
    peak_index = peaks[0]
    peak_value = output_v[peak_index]
    baseline = np.median(output_v[:max(8, len(output_v)//32)])
    output_v_normalized = (output_v - baseline) / (peak_value - baseline)
    
    # FWHM
    widths, _ = peak_widths(output_v_normalized, peaks, rel_height=0.5)
    fwhm_s = widths[0] * (time_s[1] - time_s[0])
    
    # Peak delay
    peak_delay_s = time_s[peak_index]
    
    # Output peak time
    output_peak_time_s = peak_delay_s
    
    # Energy ratio
    energy_ratio = np.trapz(output_v_normalized, time_s) / np.trapz(output_v_normalized, time_s)
    
    # Tail asymmetry
    pre_peak = output_v_normalized[:peak_index]
    post_peak = output_v_normalized[peak_index:]
    tail_energy = np.trapz(post_peak, time_s)
    pre_energy = np.trapz(pre_peak, time_s)
    tail_asymmetry = tail_energy / pre_energy if pre_energy != 0 else 0
    
    # FWHM broadening ratio
    input_fwhm_s = 0
    for j in range(len(input_v)):
        if input_v[j] > baseline:
            input_fwhm_s = time_s[j] - time_s[0]
            break
    fwhm_broadening_ratio = fwhm_s / input_fwhm_s
    
    pulse_metrics.append({
        'target_id': row['target_id'],
        'input_fwhm_s': input_fwhm_s,
        'output_fwhm_s': fwhm_s,
        'fwhm_broadening_ratio': fwhm_broadening_ratio,
        'peak_delay_s': peak_delay_s,
        'output_peak_time_s': output_peak_time_s,
        'energy_ratio': energy_ratio,
        'tail_asymmetry': tail_asymmetry
    })
pulse_metrics_df = pd.DataFrame(pulse_metrics)
pulse_metrics_df.to_csv('results/pulse_metrics.csv', index=False)
```