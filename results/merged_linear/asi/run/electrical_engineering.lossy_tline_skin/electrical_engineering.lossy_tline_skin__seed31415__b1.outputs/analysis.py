import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import resample, find_peaks, peak_widths, savgol_filter
from scipy.interpolate import interp1d

# Load input data
calibration_lines = pd.read_csv('data/calibration_lines.csv')
calibration_transfer = pd.read_csv('data/calibration_transfer.csv')
target_lines = pd.read_csv('data/target_lines.csv')
frequency_grid = pd.read_csv('data/frequency_grid.csv')
input_pulse = pd.read_csv('data/input_pulse.csv')

# Constants
c = 3e8  # Speed of light in m/s
pi = np.pi

# Fit calibration model
def objective_function(params, freq, cal_transfer, cal_lines):
    R_dc, L, G, C, k_skin, f0, k_f, k_g, k_c, k_r = params
    freq_hz = freq.values
    freq_rad = 2 * pi * freq_hz
    
    # Compute conductor loss and skin depth
    skin_depth = k_skin * np.sqrt(np.abs(freq_hz))
    Z_series = R_dc + (1 + 1j) * k_skin * np.sqrt(freq_hz) + 1j * 2 * pi * freq_hz * L
    
    # Compute shunt admittance
    Y_shunt = G + 1j * 2 * pi * freq_hz * C
    
    # Compute propagation factor
    gamma = np.sqrt(Z_series * Y_shunt)
    
    # Compute line transfer function
    H_line = np.exp(-gamma * cal_lines['length_m'])
    
    # Compute calibration transfer function
    H_cal = cal_transfer['transfer_complex'].values
    
    # Compute total transfer function
    H_total = H_cal * H_line
    
    # Compute residual
    residual = np.abs(H_total - cal_transfer['transfer_complex'].values)
    return np.sum(residual**2)

# Initial guess for parameters
initial_guess = [100, 5e-7, 0.01, 10e-12, 1e-3, 1e6, 0.5, 0.5, 0.5, 0.5]

# Optimize parameters
result = minimize(objective_function, initial_guess, args=(frequency_grid, calibration_transfer, calibration_lines))
R_dc_opt, L_opt, G_opt, C_opt, k_skin_opt, f0_opt, k_f_opt, k_g_opt, k_c_opt, k_r_opt = result.x

# Compute intrinsic line transfer function for targets
def compute_line_transfer(freq_hz, length_m, width_m, roughness):
    freq_rad = 2 * pi * freq_hz
    skin_depth = k_skin_opt * np.sqrt(np.abs(freq_hz))
    Z_series = R_dc_opt + (1 + 1j) * k_skin_opt * np.sqrt(freq_hz) + 1j * 2 * pi * freq_hz * L_opt
    
    Y_shunt = G_opt + 1j * 2 * pi * freq_hz * C_opt
    gamma = np.sqrt(Z_series * Y_shunt)
    H_line = np.exp(-gamma * length_m)
    return H_line

# Compute target transfer functions
target_transfer = []
for index, row in target_lines.iterrows():
    target_id = row['target_id']
    length_m = row['length_m']
    width_m = row['width_m']
    roughness = row['roughness']
    freq_hz = frequency_grid['frequency_hz'].values
    H_line = compute_line_transfer(freq_hz, length_m, width_m, roughness)
    magnitude_db = 20 * np.log10(np.abs(H_line))
    phase_rad = np.unwrap(np.angle(H_line))
    group_delay_s = -np.gradient(phase_rad, freq_hz) / (2 * np.pi)
    attenuation_np_per_m = 20 * np.log10(np.abs(H_line)) / length_m
    target_transfer.append({'target_id': target_id, 'frequency_hz': freq_hz, 'magnitude_db': magnitude_db, 'phase_rad': phase_rad, 'group_delay_s': group_delay_s, 'attenuation_np_per_m': attenuation_np_per_m})

target_transfer_df = pd.DataFrame(target_transfer)
target_transfer_df.to_csv('results/target_transfer.csv', index=False)

# Compute target waveforms
def compute_waveform(target_transfer, input_pulse, sampling_rate):
    freq_hz = target_transfer['frequency_hz']
    H_line = target_transfer['transfer_complex']
    freq_rad = 2 * pi * freq_hz
    H_line = H_line * np.exp(1j * 2 * pi * freq_rad * 0)
    input_pulse_freq = input_pulse['frequency_hz'].values
    input_pulse_complex = input_pulse['pulse_complex'].values
    output_pulse_complex = H_line * input_pulse_complex
    time_s = np.arange(0, len(output_pulse_complex) / sampling_rate, 1 / sampling_rate)
    output_pulse = np.real(output_pulse_complex)
    return time_s, input_pulse_complex, output_pulse

sampling_rate = 1e9  # Hz
target_waveforms = []
for index, row in target_lines.iterrows():
    target_id = row['target_id']
    freq_hz = frequency_grid['frequency_hz'].values
    H_line = compute_line_transfer(freq_hz, row['length_m'], row['width_m'], row['roughness'])
    time_s, input_pulse_complex, output_pulse = compute_waveform({'frequency_hz': freq_hz, 'transfer_complex': H_line}, input_pulse, sampling_rate)
    target_waveforms.append({'target_id': target_id, 'time_s': time_s, 'input_v': input_pulse_complex, 'output_v': output_pulse})

target_waveforms_df = pd.DataFrame(target_waveforms)
target_waveforms_df.to_csv('results/target_waveforms.csv', index=False)

# Compute pulse metrics
def compute_pulse_metrics(output_pulse, sampling_rate):
    peaks, _ = find_peaks(output_pulse, height=np.max(output_pulse) * 0.5)
    peak_indices = peaks[0]
    peak_value = output_pulse[peak_indices]
    peak_time = peak_indices / sampling_rate
    peak_delay = peak_time - (len(output_pulse) / 2) / sampling_rate
    peak_energy = np.trapz(output_pulse, dx=1 / sampling_rate)
    baseline = np.median(output_pulse[:max(8, len(output_pulse) // 32)])
    output_pulse = output_pulse - baseline
    peaks, _ = find_peaks(output_pulse, height=0.1 * np.max(output_pulse))
    peak_widths, _, _ = peak_widths(output_pulse, peaks, rel_height=0.5)
    fwhm_s = peak_widths / sampling_rate
    fwhm_ratio = fwhm_s / (len(output_pulse) / sampling_rate)
    tail_energy = np.trapz(output_pulse[peak_indices:], dx=1 / sampling_rate)
    pre_energy = np.trapz(output_pulse[:peak_indices], dx=1 / sampling_rate)
    tail_asymmetry = tail_energy / pre_energy
    return {
        'input_fwhm_s': fwhm_s[0],
        'output_fwhm_s': fwhm_s[0],
        'fwhm_broadening_ratio': fwhm_ratio[0],
        'peak_delay_s': peak_delay,
        'output_peak_time_s': peak_time,
        'energy_ratio': peak_energy / np.trapz(output_pulse, dx=1 / sampling_rate),
        'tail_asymmetry': tail_asymmetry
    }

pulse_metrics = []
for index, row in target_lines.iterrows():
    target_id = row['target_id']
    freq_hz = frequency_grid['frequency_hz'].values
    H_line = compute_line_transfer(freq_hz, row['length_m'], row['width_m'], row['roughness'])
    time_s, input_pulse_complex, output_pulse = compute_waveform({'frequency_hz': freq_hz, 'transfer_complex': H_line}, input_pulse, sampling_rate)
    metrics = compute_pulse_metrics(output_pulse, sampling_rate)
    pulse_metrics.append({'target_id': target_id, **metrics})

pulse_metrics_df = pd.DataFrame(pulse_metrics)
pulse_metrics_df.to_csv('results/pulse_metrics.csv', index=False)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(frequency_grid['frequency_hz'], target_transfer_df['magnitude_db'], label='Target Transfer Function')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.title('Target Transfer Function')
plt.legend()
plt.savefig('results/target_transfer_plot.png')

plt.figure(figsize=(12, 6))
plt.plot(frequency_grid['frequency_hz'], target_transfer_df['group_delay_s'], label='Group Delay')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Group Delay (s)')
plt.title('Group Delay vs. Frequency')
plt.legend()
plt.savefig('results/group_delay_plot.png')

plt.figure(figsize=(12, 6))
plt.plot(target_waveforms_df['time_s'], target_waveforms_df['output_v'], label='Output Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Target Output Waveform')
plt.legend()
plt.savefig('results/target_waveform_plot.png')

plt.figure(figsize=(12, 6))
plt.plot(target_waveforms_df['time_s'], target_waveforms_df['input_v'], label='Input Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Input Waveform')
plt.legend()
plt.savefig('results/input_waveform_plot.png')

plt.show()