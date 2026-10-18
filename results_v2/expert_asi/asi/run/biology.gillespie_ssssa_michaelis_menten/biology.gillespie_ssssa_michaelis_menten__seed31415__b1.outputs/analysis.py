import numpy as np
import pandas as pd
import json
from scipy.linalg import expm, inv

def load_json(path):
    with open(path) as f:
        return json.load(f)

def run_exact_cme(case_index, E_total, S_total, P_total, sample_times, observation_spec, calibration_spec):
    # Load calibration data
    calibrations = load_json('calibration_histograms.csv')
    # Compute exact transient distributions
    # ... [code for exact transient computation] ...
    # Compute stage memory probes
    # ... [code for stage memory probe] ...
    # Compute stage residual times
    # ... [code for residual time computation] ...
    # Compute stage solver codes
    # ... [code for stage solver code derivation] ...
    # Compute adaptive stage handoffs
    # ... [code for adaptive handoff computation] ...
    # Compute exact turnover summaries
    # ... [code for exact turnover summary] ...
    # Compute calibration rates
    # ... [code for calibration rate fitting] ...
    # Compute mechanism discovery
    # ... [code for mechanism selection and prediction] ...
    # Save all outputs
    # ... [code for saving outputs] ...