import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load calibration data
calib_metadata = pd.read_csv('data/calibration_samples.csv')
calib_spectra = pd.read_csv('data/calibration_spectra.csv')

# Define model families
model_families = ['bulk_only', 'size_dependent', 'size_plus_roughness']

# Evaluate each family
model_evals = []
for family in model_families:
    # Fit surrogate model
    params = fit_surrogate_model(family, calib_metadata, calib_spectra)
    
    # Cross-validate
    cv_rmse = cross_validate_model(family, calib_metadata, calib_spectra)
    
    # Evaluate held-out samples
    heldout_preds = predict_heldout_samples(family, calib_metadata, calib_spectra)
    
    # Record metrics
    model_evals.append({
        'model_name': family,
        'calibration_rmse': calib_rmse,
        'crossval_rmse': cv_rmse,
        'num_parameters': len(params),
        'preferred': family == 'bulk_only' and cv_rmse < 0.01,
        'mechanism_summary': summarize_mechanism(family)
    })

# Select preferred model
preferred_model = min(model_evals, key=lambda m: m['crossval_rmse'])['model_name']

# Predict held-out spectra
heldout_preds = predict_heldout_samples(preferred_model, calib_metadata, calib_spectra)

# Generate resonance summary
resonance_summary = generate_resonance_summary(calib_metadata, calib_spectra)

# Generate plots and diagnostics
plot_diagnostics(calib_metadata, calib_spectra, heldout_preds)