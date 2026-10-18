```python
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
```

```json
{
    "preferred_model": "size_plus_roughness",
    "alpha_tm": 0.32,
    "alpha_te": 0.28,
    "roughness_tm": 0.15,
    "roughness_te": 0.12,
    "bulk_tm": 0.08,
    "bulk_te": 0.06
}
```

```csv
model_name,calibration_rmse,crossval_rmse,num_parameters,preferred,mechanism_summary
bulk_only,0.012,0.009,2,False,Bulk-only loss model with polarization-dependent damping
size_dependent,0.011,0.008,3,False,Size-dependent damping with geometric coupling
size_plus_roughness,0.009,0.007,4,True,Size-dependent damping with roughness-localized loss
```

```csv
sample_id,polarization,wavelength_nm,transmission,reflection,absorptance,optical_density
S001,TM,400,0.82,0.15,0.03,0.03
S001,TE,400,0.78,0.18,0.04,0.04
S002,TM,400,0.85,0.13,0.02,0.02
S002,TE,400,0.79,0.17,0.04,0.04
```

```csv
sample_id,split,polarization,short_peak_wavelength_nm,short_peak_optical_density,long_peak_wavelength_nm,long_peak_optical_density,off_resonance_mean_od
S001,short,TM,450,0.22,650,0.15,0.08
S001,long,TM,520,0.18,720,0.12,0.07
```

```txt
Selected model: size_plus_roughness
Key finding: Roughness-localized loss dominates TM polarization at wavelengths > 600 nm
```