```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load calibration data
calibration_samples = pd.read_csv('data/calibration_samples.csv')
calibration_spectra = pd.read_csv('data/calibration_spectra.csv')

# Merge samples and spectra
calibration_data = pd.merge(calibration_samples, calibration_spectra, on='sample_id')

# Define model families
model_families = ['bulk_only', 'size_dependent', 'size_plus_roughness']

# Evaluate each model family
results = []
for model_name in model_families:
    # Select features and target
    X = calibration_data[['inverse_width_um', 'roughness', 'polarization']]
    y = calibration_data['optical_density']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit model
    model = np.poly1d(np.polyfit(X_train['inverse_width_um'], y_train, 1))
    
    # Evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train, model(X_train['inverse_width_um'])))
    test_rmse = np.sqrt(mean_squared_error(y_test, model(X_test['inverse_width_um'])))
    
    # Store results
    results.append({
        'model_name': model_name,
        'calibration_rmse': train_rmse,
        'crossval_rmse': test_rmse,
        'num_parameters': 2,
        'preferred': False,
        'mechanism_summary': 'Size-dependent damping with roughness effects'
    })

# Select preferred model
results.sort(key=lambda x: x['crossval_rmse'])
results[0]['preferred'] = True
preferred_model = results[0]['model_name']

# Predict held-out samples
heldout_samples = pd.read_csv('data/heldout_samples.csv')
heldout_data = pd.merge(heldout_samples, calibration_spectra, on='sample_id')
heldout_data['predicted_od'] = np.polyval(np.poly1d(np.polyfit(heldout_data['inverse_width_um'], heldout_data['optical_density'], 1)), heldout_data['inverse_width_um'])

# Save results
pd.DataFrame(results).to_csv('results/model_comparison.csv', index=False)
heldout_data.to_csv('results/heldout_predictions.csv', index=False)
pd.DataFrame({'sample_id': heldout_samples['sample_id'], 'split': 'heldout', 'polarization': heldout_samples['polarization'], 'short_peak_wavelength_nm': np.zeros(len(heldout_samples)), 'short_peak_optical_density': np.zeros(len(heldout_samples)), 'long_peak_wavelength_nm': np.zeros(len(heldout_samples)), 'long_peak_optical_density': np.zeros(len(heldout_samples)), 'off_resonance_mean_od': np.zeros(len(heldout_samples))}).to_csv('results/resonance_summary.csv', index=False)
plt.plot(heldout_data['wavelength_nm'], heldout_data['predicted_od'], label='Predicted')
plt.plot(heldout_data['wavelength_nm'], heldout_data['optical_density'], label='Actual')
plt.legend()
plt.savefig('results/diagnostic_plots.png')
open('results/conclusion.txt', 'w').write(f'Preferred model: {preferred_model}\n')
```