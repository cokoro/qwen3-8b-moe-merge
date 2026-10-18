```python
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt

def fit_model(tau, phi, a, x, R):
    return a[0] + sum(a[k] / (1 + (1j * x * tau[k]) ** phi[k]) for k in range(len(tau)))

def objective(params, x, R):
    tau, phi, a = params
    return np.sum(np.abs(R - fit_model(tau, phi, a, x, R)) ** 2)

def analyze_curves():
    data = pd.read_csv('data/curves.csv')
    x = data['x'].values
    R = data[['Re', 'Im']].values
    M = R.shape[0]
    N = R.shape[1]
    x_min, x_max = x.min(), x.max()
    x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 100)
    s = np.array([np.max([np.ptp(np.abs(R[m])), np.std(R[m, 0]), np.std(R[m, 1]), 1e-10]) for m in range(M)])
    R_normalized = R / s[:, None]
    x_grid_normalized = x_grid / x_min

    tau_guess = np.logspace(np.log10(x_min), np.log10(x_max), 10)
    phi_guess = np.full(len(tau_guess), 0.8)
    a_guess = np.full((M, len(tau_guess)), 1e-3)
    a_guess[0] = np.full(len(tau_guess), 1e-2)

    bounds = [np.log10(x_min) - 0.2, np.log10(x_max) + 0.2]
    bounds = [bounds[0], bounds[1]] + [0.72, 1.00] + [0.0, 0.0] * len(tau_guess)
    result = opt.minimize(lambda params: objective(params, x, R), bounds=bounds, method='L-BFGS-B')
    tau_opt, phi_opt, a_opt = result.x[0], result.x[1], result.x[2:]
    tau_opt = 10 ** tau_opt
    phi_opt = phi_opt
    a_opt = a_opt.reshape((M, len(tau_opt)))
    a_opt = a_opt * s[:, None]

    tau_opt = tau_opt[tau_opt.argsort()]
    phi_opt = phi_opt[tau_opt.argsort()]
    a_opt = a_opt[:, tau_opt.argsort()]

    results = pd.DataFrame({'tau': tau_opt, 'a_0': a_opt[0], 'a_1': a_opt[1], 'a_2': a_opt[2], 'a_3': a_opt[3], 'a_4': a_opt[4]})
    results.to_csv('results/modes.csv', index=False)

analyze_curves()
```