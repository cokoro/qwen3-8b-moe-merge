import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

# Step 1: Load data
K = np.load('data/K.npy')
y_obs = pd.read_csv('data/y_obs.csv').values.flatten()

# Step 2: Compute SVD
U, SVD, V = np.linalg.svd(K, full_matrices=False)
condition_number = SVD[0] / SVD[-1]

# Step 3: Compute Picard plot data
fourier_coeffs = np.abs(U.T @ y_obs)
picard_ratios = fourier_coeffs / SVD

# Step 4: Estimate noise level
noise_indices = np.argwhere(fourier_coeffs < 0.001).flatten()
noise_estimate = np.median(np.abs(fourier_coeffs[noise_indices] - np.median(fourier_coeffs[noise_indices])))

# Step 5: Determine effective rank
effective_rank = np.sum(picard_ratios > 2 * noise_estimate)

# Step 6: Write diagnosis report
diagnosis_report = {
    ...
}
with the rest of the code omitted for brevity