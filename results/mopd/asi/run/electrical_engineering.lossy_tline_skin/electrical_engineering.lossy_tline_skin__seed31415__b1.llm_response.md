```python
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load data
calibration_lines = pd.read_csv('data/calibration_lines.csv')
calibration_transfer = pd.read_csv('data/calibration_transfer.csv')
target_lines = pd.read_csv('data/target_lines.csv')
frequency_grid = pd.read_csv('data/frequency_grid.csv')
input_pulse = pd.read_csv('data/input_pulse.csv')

# Define model parameters
def model_transfer(f, R_dc, R_skin, L, G, C, length, width, roughness):
    # Convert to Hz
    f_hz = f
    # Skin effect coefficient
    k_skin = 1.0 + roughness * np.sqrt(f_hz) / (width * 1e-3)
    # Series impedance
    Z_series = R_dc + (1 + 1j) * k_skin * np.sqrt(np.abs(f_hz)) + 1j * 2 * np.pi * f_hz * L
    # Shunt admittance
    Y_shunt = G + 1j * 2 * np.pi * f_hz * C
    # Propagation constant
    gamma = np.sqrt(Z_series * Y_shunt)
    # Transfer function
    H = np.exp(-gamma * length)
    return H

# Calibration
# Fit model to calibration data
def fit_model(calibration_lines, calibration_transfer):
    # Extract calibration data
    cal_lines = calibration_lines[['length_m', 'width_m', 'roughness']]
    cal_transfer = calibration_transfer[['frequency_hz', 'magnitude_db', 'phase_rad']]
    
    # Convert to complex transfer functions
    f = cal_transfer['frequency_hz'].values
    H = np.exp(1j * cal_transfer['phase_rad'].values) * 10**(cal_transfer['magnitude_db'].values / 20)
    
    # Initial guess for model parameters
    R_dc = 1e-1
    R_skin = 1e-1
    L = 1e-7
    G = 1e-3
    C = 1e-12
    
    # Optimize parameters
    from scipy.optimize import minimize
    def objective(params):
        R_dc, R_skin, L, G, C = params
        # Compute model transfer function
        H_model = np.zeros_like(H, dtype=complex)
        for i in range(len(f)):
            H_model[i] = model_transfer(f[i], R_dc, R_skin, L, G, C, cal_lines['length_m'][i], cal_lines['width_m'][i], cal_lines['roughness'][i])
        # Compute error
        error = np.sum(np.abs(H_model - H)**2)
        return error
    
    result = minimize(objective, [R_dc, R_skin, L, G, C], method='L-BFGS-B')
    R_dc, R_skin, L, G, C = result.x
    
    # Return optimized parameters
    return R_dc, R_skin, L, G, C

# Predict target transfer functions
def predict_transfers(target_lines, frequency_grid, R_dc, R_skin, L, G, C):
    # Extract target data
    target_ids = target_lines['target_id'].values
    target_lengths = target_lines['length_m'].values
    target_widths = target_lines['width_m'].values
    target_roughness = target_lines['roughness'].values
    
    # Convert to complex transfer functions
    f = frequency_grid['frequency_hz'].values
    H = np.zeros((len(target_ids), len(f)), dtype=complex)
    
    # Compute transfer function for each target
    for i in range(len(target_ids)):
        H[i, :] = model_transfer(f, R_dc, R_skin, L, G, C, target_lengths[i], target_widths[i], target_roughness[i])
    
    # Convert to magnitude and phase
    magnitude_db = 20 * np.log10(np.abs(H))
    phase_rad = np.unwrap(np.angle(H))
    group_delay_s = -np.gradient(phase_rad, f) / (2 * np.pi)
    attenuation_np_per_m = 20 * np.log10(np.abs(H)) / (target_lengths * 1e3)
    
    # Return results
    return pd.DataFrame({
        'target_id': target_ids,
        'frequency_hz': f,
        'magnitude_db': magnitude_db,
        'phase_rad': phase_rad,
        'group_delay_s': group_delay_s,
        'attenuation_np_per_m': attenuation_np_per_m
    })

# Generate waveforms
def generate_waveforms(target_transfer, input_pulse):
    # Extract target data
    target_ids = target_transfer['target_id'].values
    f = target_transfer['frequency_hz'].values
    H = target_transfer['magnitude_db'].values + 1j * target_transfer['phase_rad'].values
    H = np.exp(1j * H) * 10**(H / 20)
    
    # Generate time samples
    t = np.linspace(0, 1, len(input_pulse['time_s']))
    input_v = input_pulse['input_v'].values
    output_v = np.zeros_like(input_v)
    
    # Compute waveform for each target
    for i in range(len(target_ids)):
        # Compute output signal
        output_v = input_v * H[i]
    
    # Return results
    return pd.DataFrame({
        'target_id': target_ids,
        'time_s': t,
        'input_v': input_v,
        'output_v': output_v
    })

# Compute pulse metrics
def compute_metrics(target_transfer, target_waveforms):
    # Extract target data
    target_ids = target_transfer['target_id'].values
    t = target_waveforms['time_s'].values
    input_v = target_waveforms['input_v'].values
    output_v = target_waveforms['output_v'].values
    
    # Compute metrics
    metrics = []
    for i in range(len(target_ids)):
        # Compute FWHM
        peaks, _ = find_peaks(output_v[i], height=0.5 * np.max(output_v[i]))
        if len(peaks) == 0:
            fwhm = np.nan
        else:
            fwhm = t[peaks[0]] - t[peaks[-1]]
        # Compute energy ratio
        baseline = np.median(output_v[i][:max(8, len(output_v[i])//32)])
        energy = np.trapz(np.abs(output_v[i] - baseline), t)
        energy_ratio = energy / np.trapz(input_v - baseline, t)
        # Compute tail asymmetry
        peak_idx = np.argmax(output_v[i])
        pre_peak = output_v[i][:peak_idx]
        post_peak = output_v[i][peak_idx:]
        tail_asymmetry = np.sum(post_peak) / np.sum(pre_peak)
        # Compute peak delay
        peak_time = t[peak_idx]
        peak_delay = peak_time - t[0]
        # Compute metrics
        metrics.append({
            'target_id': target_ids[i],
            'input_fwhm_s': np.nan,
            'output_fwhm_s': fwhm,
            'fwhm_broadening_ratio': np.nan,
            'peak_delay_s': peak_delay,
            'output_peak_time_s': peak_time,
            'energy_ratio': energy_ratio,
            'tail_asymmetry': tail_asymmetry
        })
    
    # Return results
    return pd.DataFrame(metrics)

# Run analysis
R_dc, R_skin, L, G, C = fit_model(calibration_lines, calibration_transfer)
target_transfer = predict_transfers(target_lines, frequency_grid, R_dc, R_skin, L, G, C)
target_waveforms = generate_waveforms(target_transfer, input_pulse)
pulse_metrics = compute_metrics(target_transfer, target_waveforms)

# Save results
target_transfer.to_csv('results/target_transfer.csv', index=False)
target_waveforms.to_csv('results/target_waveforms.csv', index=False)
pulse_metrics.to_csv('results/pulse_metrics.csv', index=False)
```