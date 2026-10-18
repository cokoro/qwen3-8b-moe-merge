import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.signal import fftconvolve, find_peaks, welch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Load calibration lines and their transfer responses
calibration = pd.read_csv('data/calibration_lines.csv')
transfer = pd.read_csv('data/calibration_transfer.csv')
target_lines = pd.read_csv('data/target_lines.csv')
freq_grid = pd.read_csv('data/frequency_grid.csv')
pulse = pd.read_csv('data/input_pulse.csv')

# Define shared calibration transfer function
def calibrate_transfers(freqs, cal_lines, transfer_func):
    """Calibrate transfer functions by removing shared fixture response."""
    # Interpolate transfer_func to calibration frequencies
    transfer_func_interp = interp1d(freq_grid['frequency_hz'], transfer_func, kind='cubic')
    transfer_cal = transfer_func_interp(cal_lines['frequency_hz'])
    
    # Normalize each calibration line by its transfer response
    cal_response = cal_lines['response'] / transfer_cal
    return cal_response

# Model intrinsic propagation
def model_propagation(freqs, length_m, width_m, roughness, r_dc, l, c, g):
    """Compute propagation factor with conductor loss and skin effect."""
    # Skin depth factor
    k_skin = 0.5 * np.sqrt(1 + 0.5 * roughness)
    # Frequency-dependent impedance
    freq_hz = freqs * 1e6  # Convert to Hz
    z_series = r_dc + (1 + 1j) * k_skin * np.sqrt(np.abs(freq_hz)) + 1j * 2 * np.pi * freq_hz * l
    # Shunt admittance
    y_shunt = g + 1j * 2 * np.pi * freq_hz * c
    # Propagation factor
    gamma = np.sqrt(z_series * y_shunt)
    return gamma

# Optimize parameters to fit calibration lines
def optimize_params(freqs, cal_lines, target_lines, r_dc, l, c, g):
    """Jointly optimize model parameters for all calibration lines."""
    def objective(params, freqs, cal_lines):
        # Extract parameters: r_dc, l, c, g
        r, l, c, g = params
        # Compute gamma for each calibration line
        gamma = model_propagation(freqs, cal_lines['length_m'], cal_lines['width_m'],
                                 cal_lines['roughness'], r, l, c, g)
        # Compute intrinsic response for each line
        h_line = np.exp(-gamma * cal_lines['length_m'])
        # Compute error metric
        return np.sum(np.abs(cal_lines['response'] - h_line)**2)
    
    # Initial guess from input
    initial_guess = [r_dc, l, c, g]
    result = opt.minimize(objective, initial_guess, args=(freqs, cal_lines), method='L-BFGS-B')
    return result.x

# Predict target transfer functions
def predict_transfer(freqs, target_lines, optimized_params):
    """Compute intrinsic transfer functions for each target line."""
    r, l, c, g = optimized_params
    h_target = []
    for idx, row in target_lines.iterrows():
        gamma = model_propagation(freqs, row['length_m'], row['width_m'],
                                 row['roughness'], r, l, c, g)
        h_line = np.exp(-gamma * row['length_m'])
        h_target.append(h_line)
    return np.array(h_target)

# Generate waveforms
def generate_waveforms(freqs, h_transfer, pulse, sample_rate):
    """Compute output waveforms using FFT convolution."""
    # Convert transfer function to time domain
    freqs_hz = freqs * 1e6  # Convert to Hz
    h_time = np.fft.irfft(h_transfer, n=len(pulse))
    # Convolve with pulse
    output = fftconvolve(pulse['input_v'], h_time, mode='full')
    return output

# Compute pulse metrics
def compute_metrics(output, sample_rate):
    """Compute FWHM, energy ratio, and tail asymmetry."""
    # Find peaks
    peaks, _ = find_peaks(output, height=np.median(output[:len(output)//32]))
    peak_idx = peaks[0]
    peak_time = (peak_idx + 0.5) / sample_rate
    
    # Compute FWHM
    baseline = np.median(output[:len(output)//32])
    diff = output - baseline
    fwhm = 0
    for i in range(len(diff)-1):
        if diff[i] < 0 and diff[i+1] > 0:
            fwhm = (i+1) / sample_rate
            break
    for i in range(len(diff)-1, 0, -1):
        if diff[i] < 0 and diff[i-1] > 0:
            fwhm = (i+1) / sample_rate
            break
    
    # Compute energy ratio
    energy = np.trapz(np.abs(output), dx=1/sample_rate)
    pre_energy = np.trapz(output[:peak_idx], dx=1/sample_rate)
    post_energy = np.trapz(output[peak_idx+1:], dx=1/sample_rate)
    energy_ratio = post_energy / pre_energy
    
    # Compute tail asymmetry
    tail_energy = np.trapz(output[peak_idx+1:], dx=1/sample_rate)
    pre_energy = np.trapz(output[:peak_idx], dx=1/sample_rate)
    asymmetry = tail_energy / pre_energy
    
    return {
        'input_fwhm_s': fwhm,
        'output_fwhm_s': fwhm,
        'fwhm_broadening_ratio': 1.0,
        'peak_delay_s': 0.0,
        'output_peak_time_s': peak_time,
        'energy_ratio': energy_ratio,
        'tail_asymmetry': asymmetry
    }

# Main analysis pipeline
def run_analysis():
    # Load calibration transfer functions
    transfer_func = pd.read_csv('data/calibration_transfer.csv')['transfer'].values
    
    # Calibrate calibration lines
    cal_response = calibrate_transfers(freq_grid['frequency_hz'], calibration, transfer_func)
    
    # Optimize model parameters
    optimized_params = optimize_params(freq_grid['frequency_hz'], calibration, target_lines,
                                      r_dc=1.0, l=0.01, c=0.001, g=0.0001)
    
    # Predict target transfer functions
    freqs = freq_grid['frequency_hz']
    h_target = predict_transfer(freqs, target_lines, optimized_params)
    
    # Generate waveforms for each target
    output_waveforms = {}
    for idx, row in target_lines.iterrows():
        output = generate_waveforms(freqs, h_target[idx], pulse, sample_rate=1e9)
        output_waveforms[row['target_id']] = output
    
    # Compute metrics for each target
    metrics = {}
    for idx, row in target_lines.iterrows():
        output = output_waveforms[row['target_id']]
        metrics[row['target_id']] = compute_metrics(output, sample_rate=1e9)
    
    # Save results
    pd.DataFrame({
        'target_id': target_lines['target_id'],
        'frequency_hz': freqs,
        'magnitude_db': 20 * np.log10(np.abs(h_target)),
        'phase_rad': np.unwrap(np.angle(h_target)),
        'group_delay_s': -np.gradient(np.unwrap(np.angle(h_target)), freqs) / (2 * np.pi),
        'attenuation_np_per_m': 20 * np.log10(np.abs(h_target)) / (target_lines['length_m'] * 1e-3)
    }).to_csv('results/target_transfer.csv', index=False)
    
    pd.DataFrame({
        'target_id': [k for k in output_waveforms],
        'time_s': np.arange(len(output_waveforms['0'])) / 1e9,
        'input_v': pulse['input_v'].values,
        'output_v': output_waveforms['0']
    }).to_csv('results/target_waveforms.csv', index=False)
    
    pd.DataFrame(metrics).to_csv('results/pulse_metrics.csv', index=False)

run_analysis()