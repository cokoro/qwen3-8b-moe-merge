import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds

def fit_model(x, y_real, y_imag, ell, phi, a, s_m, a0):
    # Compute model prediction
    R = a0 + np.sum(a / (1 + (1j * x)**phi * 10**ell), axis=0)
    # Compute residuals
    residual_real = (y_real - R.real) / s_m
    residual_imag = (y_imag - R.imag) / s_m
    return np.sum(residual_real**2 + residual_imag**2)

def objective(params, x, y_real, y_imag, s_m, K):
    ell = params[:K]
    phi = params[K:2*K]
    a = params[2*K:2*K + K]
    a0 = params[-1]
    return fit_model(x, y_real, y_imag, ell, phi, a, s_m, a0)

def main():
    df = pd.read_csv('data/curves.csv')
    x = df['x'].values
    y_real = df['real'].values
    y_imag = df['imag'].values
    M = len(y_real)
    N = len(x)
    
    # Normalize each curve
    s_m = np.max(np.array([np.ptp(np.abs(y_real)), np.std(y_real), np.std(y_imag), 1e-10]), axis=0)
    
    # Initial guess: use simple linear fit
    # Use a grid search for ell, phi, and a
    K = 5
    bounds = Bounds([-np.log10(x.max()) - 0.2, -np.log10(x.min()) - 0.2], 
                   [np.log10(x.max()) - 0.2, np.log10(x.min()) + 0.2], 
                   keep_feasible=True)
    phi_bounds = Bounds([0.72, 0.72, 0.72, 0.72, 0.72], [1.0, 1.0, 1.0, 1.0, 1.0])
    
    # Initialize parameters with a grid search
    # Here, we use a simplified approach for demonstration purposes
    # In practice, a more sophisticated search would be used
    
    # Initial guess for ell
    ell_guess = np.linspace(np.log10(x.min()) + 0.2, np.log10(x.max()) - 0.2, K)
    # Initial guess for phi
    phi_guess = np.full(K, 0.8)
    # Initial guess for a
    a_guess = np.ones(K) * 0.1
    # Initial guess for a0
    a0_guess = np.mean(y_real)
    
    # Combine parameters
    initial_params = np.concatenate([ell_guess, phi_guess, a_guess, [a0_guess]])
    
    # Optimize
    result = minimize(objective, initial_params, args=(x, y_real, y_imag, s_m, K),
                      bounds=Bounds([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
                                    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
                      method='L-BFGS-B', tol=1e-8)
    
    # Save results
    tau = np.exp(10**result.x[:K])
    a0 = result.x[-1]
    a = result.x[2*K:2*K + K]
    
    # Output results
    df_results = pd.DataFrame({
        'tau': tau,
        'a0': a0,
        'a1': a[0],
        'a2': a[1],
        'a3': a[2],
        'a4': a[3],
        'a5': a[4]
    })
    df_results.to_csv('results/modes.csv', index=False)

if __name__ == '__main__':
    main()