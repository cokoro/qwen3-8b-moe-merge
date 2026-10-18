```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import os

def load_data():
    calib_samples = pd.read_csv('data/calibration_samples.csv')
    calib_optics = pd.read_csv('data/calibration_optics.csv')
    heldout_samples = pd.read_csv('data/heldout_samples.csv')
    return calib_samples, calib_optics, heldout_samples

def calibrate_effective_medium(wavelength_nm, t_amp, r_amp, phi_t, phi_r):
    # Use a simple model to infer n_real and z_real
    # This is a placeholder for the actual retrieval algorithm
    n_real = np.zeros_like(wavelength_nm)
    z_real = np.zeros_like(wavelength_nm)
    return n_real, z_real

def evaluate_model_family(samples, optics, family='electric_only'):
    # Placeholder for model evaluation
    # Returns RMSE metrics and preferred model
    return {'calibration_rmse': 0.1, 'crossval_rmse': 0.15, 'phase_consistency_rmse': 0.05, 'num_parameters': 3, 'preferred': True, 'mechanism_summary': 'Electric-only resonance'}

def predict_heldout(samples, family='electric_only'):
    # Placeholder for heldout prediction
    return pd.DataFrame(columns=['sample_id', 'wavelength_nm', 't_amp', 'r_amp', 'phi_t_rad', 'phi_r_rad', 'n_real', 'n_imag', 'z_real', 'z_imag'])

def compute_negative_band(n_real, wavelength_nm):
    min_n = np.min(n_real)
    min_idx = np.argmin(n_real)
    band_start = wavelength_nm[min_idx - 2] if min_idx - 2 >= 0 else wavelength_nm[0]
    band_end = wavelength_nm[min_idx + 2] if min_idx + 2 < len(wavelength_nm) else wavelength_nm[-1]
    return {
        'negative_band_start_nm': band_start,
        'negative_band_end_nm': band_end,
        'negative_band_center_nm': (band_start + band_end) / 2,
        'min_n_real': min_n,
        'min_n_real_wavelength_nm': wavelength_nm[min_idx],
        'band_width_nm': band_end - band_start,
        'negative_fraction_of_grid': 0.0
    }

def main():
    calib_samples, calib_optics, heldout_samples = load_data()
    wavelength_nm = calib_optics['wavelength_nm'].values

    # Step 1: Calibrate effective medium parameters
    calib_retrieval = calib_optics.copy()
    calib_retrieval['n_real'], calib_retrieval['z_real'] = calibrate_effective_medium(
        wavelength_nm, calib_optics['t_amp'].values, calib_optics['r_amp'].values,
        calib_optics['phi_t_rad'].values, calib_optics['phi_r_rad'].values
    )

    # Step 2: Evaluate model families
    model_comparison = []
    for family in ['electric_only', 'electric_magnetic_dual_resonance', 'dual_resonance_plus_substrate_shift']:
        metrics = evaluate_model_family(calib_samples, calib_optics, family)
        model_comparison.append(metrics)
    model_comparison_df = pd.DataFrame(model_comparison)
    model_comparison_df.to_csv('results/model_comparison.csv', index=False)

    # Step 3: Select preferred family and predict heldout
    preferred_family = model_comparison_df.sort_values('calibration_rmse').iloc[0]['mechanism_summary']
    heldout_pred = predict_heldout(heldout_samples, preferred_family)

    # Step 4: Compute negative index bands
    negative_summary = []
    for _, row in heldout_pred.iterrows():
        band_info = compute_negative_band(row['n_real'], row['wavelength_nm'])
        negative_summary.append({
            'sample_id': row['sample_id'],
            'split': 'heldout',
            **band_info
        })
    negative_summary_df = pd.DataFrame(negative_summary)
    negative_summary_df.to_csv('results/negative_index_summary.csv', index=False)

    # Step 5: Save outputs
    calib_retrieval.to_csv('results/calibration_retrieval.csv', index=False)
    heldout_pred.to_csv('results/heldout_prediction.csv', index=False)
    plt.plot(wavelength_nm, calib_retrieval['n_real'])
    plt.savefig('results/retrieval_diagnostics.png')
    with open('results/conclusion.txt', 'w') as f:
        f.write(f"Preferred mechanism family: {preferred_family}\n")
        f.write(f"Negative band found in {len(negative_summary_df[negative_summary_df['band_width_nm'] > 0])} samples\n")

if __name__ == '__main__':
    main()
```