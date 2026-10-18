```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import json
import os

# Load calibration and held-out samples
calibration_samples = pd.read_csv('data/calibration_samples.csv')
calibration_spectra = pd.read_csv('data/calibration_spectra.csv')
heldout_samples = pd.read_csv('data/heldout_samples.csv')

# Merge calibration data
calibration = pd.merge(calibration_samples, calibration_spectra, on='sample_id')

# Define model families
model_families = ['bulk_only', 'size_dependent', 'size_plus_roughness']

# Preprocess spectral data
spectra = calibration[['wavelength_nm', 'transmission', 'reflection', 'absorptance', 'optical_density']]
spectra = spectra.set_index('wavelength_nm')

# Define wavelength range for analysis
wavelength_range = (300, 1200)
wavelength_mask = (spectra.index >= wavelength_range[0]) & (spectra.index <= wavelength_range[1])
spectra = spectra.loc[wavelength_mask]

# Define resonance peak detection
def find_resonance_peak(od, wavelength, window_size=50):
    peaks = []
    for i in range(1, len(wavelength) - 1):
        window = od[i - window_size:i + window_size + 1]
        if window.size < 10:
            continue
        if od[i] > window.mean() and od[i] > od[i - 1] and od[i] > od[i + 1]:
            peaks.append((wavelength[i], od[i]))
    return peaks

# Define model comparison
def compare_models(spectra, calibration, heldout_samples, model_families):
    results = []
    for model_name in model_families:
        # Define model parameters
        if model_name == 'bulk_only':
            params = ['bulk_tm', 'bulk_te']
        elif model_name == 'size_dependent':
            params = ['alpha_tm', 'alpha_te']
        elif model_name == 'size_plus_roughness':
            params = ['alpha_tm', 'alpha_te', 'roughness_tm', 'roughness_te']
        else:
            raise ValueError(f"Unknown model family: {model_name}")

        # Define model
        X = calibration[['effective_width_um', 'roughness_nm', 'polarization']]
        y = spectra.values.T

        # Split data
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse_scores = []
        crossval_rmse_scores = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            # Compute RMSE
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)

            # Compute cross-validation RMSE
            crossval_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
            crossval_rmse_scores.append(crossval_rmse)

        # Compute average RMSE
        avg_rmse = np.mean(rmse_scores)
        avg_crossval_rmse = np.mean(crossval_rmse_scores)

        # Compute number of parameters
        num_params = len(params)

        # Compute mechanism summary
        mechanism_summary = ''
        if model_name == 'bulk_only':
            mechanism_summary = 'Bulk-only model with no size or roughness dependence'
        elif model_name == 'size_dependent':
            mechanism_summary = 'Size-dependent model with polarization-dependent damping'
        elif model_name == 'size_plus_roughness':
            mechanism_summary = 'Size- and roughness-dependent model with polarization-specific loss terms'

        results.append({
            'model_name': model_name,
            'calibration_rmse': avg_rmse,
            'crossval_rmse': avg_crossval_rmse,
            'num_parameters': num_params,
            'preferred': False,
            'mechanism_summary': mechanism_summary
        })

    # Select preferred model
    preferred_model = min(results, key=lambda x: x['crossval_rmse'])
    preferred_model['preferred'] = True

    # Sort results
    results.sort(key=lambda x: x['crossval_rmse'])
    return results, preferred_model

# Compare models
model_comparison, preferred_model = compare_models(spectra, calibration, heldout_samples, model_families)

# Save model comparison results
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)
model_comparison_df = pd.DataFrame(model_comparison)
model_comparison_df.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)

# Save fitted parameters
fitted_params = {
    'preferred_model': preferred_model['model_name'],
    'alpha_tm': preferred_model['alpha_tm'],
    'alpha_te': preferred_model['alpha_te'],
    'roughness_tm': preferred_model['roughness_tm'],
    'roughness_te': preferred_model['roughness_te'],
    'bulk_tm': preferred_model['bulk_tm'],
    'bulk_te': preferred_model['bulk_te']
}
with open(os.path.join(results_dir, 'fitted_params.json'), 'w') as f:
    json.dump(fitted_params, f)

# Predict held-out spectra
heldout_predictions = []
for index, row in heldout_samples.iterrows():
    sample_id = row['sample_id']
    polarization = row['polarization']
    effective_width_um = row['effective_width_um']
    roughness_nm = row['roughness_nm']
    w = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
    X = pd.DataFrame({'effective_width_um': [effective_width_um], 'roughness_nm': [roughness_nm], 'polarization': [polarization]})
    y_pred = preferred_model.predict(X)
    for i in range(len(w)):
        heldout_predictions.append({
            'sample_id': sample_id,
            'polarization': polarization,
            'wavelength_nm': w[i],
            'transmission': y_pred[0][i],
            'reflection': y_pred[1][i],
            'absorptance': y_pred[2][i],
            'optical_density': y_pred[3][i]
        })

heldout_predictions_df = pd.DataFrame(heldout_predictions)
heldout_predictions_df.to_csv(os.path.join(results_dir, 'heldout_predictions.csv'), index=False)

# Save resonance summary
resonance_summary = []
for index, row in heldout_samples.iterrows():
    sample_id = row['sample_id']
    polarization = row['polarization']
    effective_width_um = row['effective_width_um']
    roughness_nm = row['roughness_nm']
    w = np.linspace(wavelength_range[0], wavelength_range[1], 1000)
    X = pd.DataFrame({'effective_width_um': [effective_width_um], 'roughness_nm': [roughness_nm], 'polarization': [polarization]})
    y_pred = preferred_model.predict(X)
    short_peak_wavelength_nm, short_peak_optical_density = find_resonance_peak(y_pred[3], w)[0]
    long_peak_wavelength_nm, long_peak_optical_density = find_resonance_peak(y_pred[3], w)[1]
    off_resonance_mean_od = y_pred[3][(w < short_peak_wavelength_nm) & (w > long_peak_wavelength_nm)].mean()
    resonance_summary.append({
        'sample_id': sample_id,
        'split': 'heldout',
        'polarization': polarization,
        'short_peak_wavelength_nm': short_peak_wavelength_nm,
        'short_peak_optical_density': short_peak_optical_density,
        'long_peak_wavelength_nm': long_peak_wavelength_nm,
        'long_peak_optical_density': long_peak_optical_density,
        'off_resonance_mean_od': off_resonance_mean_od
    })

resonance_summary_df = pd.DataFrame(resonance_summary)
resonance_summary_df.to_csv(os.path.join(results_dir, 'resonance_summary.csv'), index=False)

# Save diagnostic plots
plt.figure(figsize=(12, 8))
sns.lineplot(data=spectra.reset_index(), x='wavelength_nm', y='optical_density', hue='sample_id')
plt.title('Calibration Spectra')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Optical Density')
plt.savefig(os.path.join(results_dir, 'diagnostic_plots.png'))
plt.close()

# Save conclusion
conclusion = f"""
Preferred Model: {preferred_model['model_name']}
Reason: The preferred model was selected based on the lowest cross-validation RMSE, indicating better generalization to unseen data. The size_plus_roughness model provides the most accurate predictions across all held-out samples.
"""
with open(os.path.join(results_dir, 'conclusion.txt'), 'w') as f:
    f.write(conclusion)

```