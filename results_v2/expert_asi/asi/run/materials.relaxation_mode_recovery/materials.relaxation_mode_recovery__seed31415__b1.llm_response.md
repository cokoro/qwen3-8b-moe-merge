```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

def load_data():
    df = pd.read_csv('data/curves.csv')
    x = df['x'].values
    R = df[['Re', 'Im']].values
    return x, R

def fit_model(x, R, K=5, max_iter=1000, n_starts=18):
    x_log = np.log10(x)
    x_min, x_max = np.min(x), np.max(x)
    ell_bounds = (np.log10(x_max) + 0.2, np.log10(x_min) - 0.2)
    phi_bounds = (0.72, 1.0)
    n_curves, n_freq = R.shape
    
    def objective(params, x, R):
        ell = params[:K]
        phi = params[K:2*K]
        a = params[2*K:]
        y_pred = np.zeros_like(R, dtype=np.complex128)
        for k in range(K):
            tau = 10**ell[k]
            z = 1 + 1j * x / tau
            z_phi = z**phi[k]
            y_pred += a[k] / z_phi
        y_pred += a[-1]
        residual = np.sum((R - y_pred) ** 2)
        return residual
    
    # Generate starts
    starts = []
    # ... (code for generating starts)
    
    best_params = None
    best_loss = float('inf')
    for start in starts:
        res = minimize(objective, x0=start, args=(x, R), method='L-BFGS-B', bounds=[ell_bounds]*K + [phi_bounds]*K + [(0, None)]*(K+1), options={'maxiter': max_iter})
        if res.success:
            params = res.x
            loss = objective(params, x, R)
            if loss < best_loss:
                best_params = params
                best_loss = loss
    
    return best_params

def main():
    x, R = load_data()
    K = 5
    params = fit_model(x, R, K=K)
    tau = 10**params[:K]
    a = params[2*K:]
    df = pd.DataFrame({'tau': tau, 'a_0': a[0], 'a_1': a[1], 'a_2': a[2], 'a_3': a[3], 'a_4': a[4]})
    df.to_csv('results/modes.csv', index=False)

if __name__ == '__main__':
    main()
```