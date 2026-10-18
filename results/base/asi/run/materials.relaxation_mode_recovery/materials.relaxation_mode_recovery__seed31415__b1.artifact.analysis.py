import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import sys

def load_data():
    df = pd.read_csv('data/curves.csv')
    freqs = df['freq'].values
    real = df['real'].values
    imag = df['imag'].values
    return freqs, real, imag

def model(x, tau, phi, a, a0):
    return a0 + np.sum(a / (1 + (1j * x * tau) ** phi), axis=1)

def objective(params, x, y_real, y_imag, K, freqs):
    tau = np.log10(params[::K+1])
    phi = params[1::K+1]
    a = params[2::K+1]
    a0 = params[-1]
    pred = model(freqs, tau, phi, a, a0)
    residuals = np.concatenate((pred.real - y_real, pred.imag - y_imag))
    return np.sum(residuals ** 2)

def normalize_data(y_real, y_imag):
    s = np.maximum(np.ptp(np.abs(y_real)), np.std(y_real), np.std(y_imag), 1e-10)
    return y_real / s, y_imag / s

def generate_starts(freqs, K, real, imag):
    starts = []
    # Generate initial guesses based on linear fits
    for phi in [0.78, 0.86, 0.94]:
        # Normalize data
        y_real_norm, y_imag_norm = normalize_data(real, imag)
        # Fit linear model
        A = np.vstack([np.log10(freqs), np.ones_like(freqs)]).T
        coeffs = np.linalg.lstsq(A, np.concatenate((y_real_norm, y_imag_norm)), rcond=None)[0]
        # Generate initial tau guesses
        ell = np.linspace(np.log10(freqs.min()), np.log10(freqs.max()), 96)
        tau = 10 ** ell
        # Generate initial a and a0
        a = np.random.rand(K)
        a0 = np.random.rand()
        starts.append((tau, phi, a, a0))
    return starts

def fit_model(freqs, real, imag, K, starts, max_iter=1000):
    best_params = None
    best_rss = np.inf
    for start in starts:
        tau_start, phi_start, a_start, a0_start = start
        # Initial guess
        initial_guess = np.concatenate((np.log10(tau_start), phi_start, a_start, [a0_start]))
        # Optimization
        result = minimize(objective, initial_guess, args=(freqs, real, imag, K, freqs), method='L-BFGS-B', options={'maxiter': max_iter})
        if result.success:
            params = result.x
            tau = 10 ** params[::K+1]
            phi = params[1::K+1]
            a = params[2::K+1]
            a0 = params[-1]
            pred = model(freqs, tau, phi, a, a0)
            rss = np.sum((pred.real - real) ** 2 + (pred.imag - imag) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_params = params
    return best_params

def calculate_likelihood_loss(params, freqs, real, imag, K):
    tau = 10 ** params[::K+1]
    phi = params[1::K+1]
    a = params[2::K+1]
    a0 = params[-1]
    pred = model(freqs, tau, phi, a, a0)
    rss = np.sum((pred.real - real) ** 2 + (pred.imag - imag) ** 2)
    n = 2 * len(freqs) * len(real)
    rss_min = np.min(rss)
    L = max(0, rss - rss_min) / max(rss_min / n, 1e-12)
    return L

def calculate_regularization(params, freqs, real, imag, K):
    tau = 10 ** params[::K+1]
    phi = params[1::K+1]
    a = params[2::K+1]
    a0 = params[-1]
    # Duplicate penalty
    g = np.zeros(K)
    for i in range(K):
        for j in range(i+1, K):
            g[i] += np.abs(np.log10(tau[i]) - np.log10(tau[j]))
    P_duplicate = 16 * np.sum(np.maximum(0, 0.14 - g) / 0.14) ** 2
    # Support penalty
    u = np.zeros(K)
    for i in range(K):
        u[i] = np.linalg.norm(a[i] / np.std(real, axis=0), 2)
    P_support = 10 * np.sum(np.maximum(0, 0.012 - u) / 0.012) ** 2
    # Isolated penalty
    r = np.zeros(K)
    for i in range(K):
        r[i] = a[i] / np.maximum(np.median(a), 0.08 * np.std(real, axis=0))
    q = np.max(r, axis=1)
    P_isolated = 4 * np.sum((u < 0.025) & (q < 0.35))
    # Phi penalty
    P_phi = 0.25 * np.sum(
        np.maximum(0, 0.74 - phi) / 0.04 ** 2 +
        np.maximum(0, phi - 0.99) / 0.04 ** 2
    )
    return P_duplicate + P_support + P_isolated + P_phi

def main():
    freqs, real, imag = load_data()
    K = 5
    starts = generate_starts(freqs, K, real, imag)
    best_params = fit_model(freqs, real, imag, K, starts)
    # Calculate likelihood loss and regularization
    L = calculate_likelihood_loss(best_params, freqs, real, imag, K)
    P = calculate_regularization(best_params, freqs, real, imag, K)
    J = L + P
    # Output results
    tau = 10 ** best_params[::K+1]
    phi = best_params[1::K+1]
    a = best_params[2::K+1]
    a0 = best_params[-1]
    df = pd.DataFrame({
        'tau': tau,
        'a_0': a0,
        'a_1': a[0],
        'a_2': a[1],
        'a_3': a[2],
        'a_4': a[3],
        'a_5': a[4]
    })
    df.to_csv('results/modes.csv', index=False)

if __name__ == '__main__':
    main()