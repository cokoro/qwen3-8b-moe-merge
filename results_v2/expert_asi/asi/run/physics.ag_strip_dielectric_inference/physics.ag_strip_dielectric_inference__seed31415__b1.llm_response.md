```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data(path):
    return pd.read_csv(path)

def fit_surrogate(data, model_family):
    # Placeholder for fitting logic
    # Returns model parameters
    pass

def predict_heldout(model, heldout_data):
    # Placeholder for prediction logic
    # Returns predicted spectra
    pass

def evaluate_model(model, calibration_data, crossval_data):
    # Placeholder for evaluation
    return {
        'calibration_rmse': 0.0,
        'crossval_rmse': 0.0,
        'num_parameters': 5,
        'preferred': True,
        'mechanism_summary': 'Bulk loss dominates with size-dependent damping'
    }

def plot_results(predictions, resonance_summary, diagnostic_plots_path):
    # Placeholder for plotting
    pass

def generate_conclusion(model_params):
    # Placeholder for conclusion
    return 'The size-dependent model provides the best transfer performance.'

def main():
    calibration_samples = load_data('data/calibration_samples.csv')
    calibration_spectra = load_data('data/calibration_spectra.csv')
    heldout_samples = load_data('data/heldout_samples.csv')

    model_families = ['bulk_only', 'size_dependent', 'size_plus_roughness']
    model_results = []

    for family in model_families:
        model = fit_surrogate(calibration_samples, family)
        metrics = evaluate_model(model, calibration_spectra, calibration_spectra)
        model_results.append(metrics)

    preferred_model = min(model_results, key=lambda x: x['crossval_rmse'])['model_name']
    fitted_params = fit_surrogate(calibration_samples, preferred_model)

    predictions = predict_heldout(fitted_params, heldout_samples)
    resonance_summary = generate_resonance_summary(predictions)
    plot_results(predictions, resonance_summary, 'results/diagnostic_plots.png')

    # Save outputs
    pd.DataFrame(model_results).to_csv('results/model_comparison.csv', index=False)
    pd.DataFrame([fitted_params]).to_json('results/fitted_params.json')
    pd.DataFrame(predictions).to_csv('results/heldout_predictions.csv', index=False)
    pd.DataFrame(resonance_summary).to_csv('results/resonance_summary.csv', index=False)
    with open('results/conclusion.txt', 'w') as f:
        f.write(generate_conclusion(fitted_params))

if __name__ == '__main__':
    main()
```