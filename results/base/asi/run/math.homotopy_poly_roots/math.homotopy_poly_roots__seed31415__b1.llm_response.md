```python
import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import json
import os

def parse_polynomial(terms):
    def evaluate(coeffs, x0, x1):
        result = 0.0
        for term in coeffs:
            coeff = complex(term['coeff_re'], term['coeff_im'])
            exp0 = term['exponents'][0]
            exp1 = term['exponents'][1]
            result += coeff * (x0 ** exp0) * (x1 ** exp1)
        return result
    return evaluate

def build_polynomial(terms):
    def evaluate(x0, x1):
        result = 0.0
        for term in terms:
            coeff = complex(term['coeff_re'], term['coeff_im'])
            exp0 = term['exponents'][0]
            exp1 = term['exponents'][1]
            result += coeff * (x0 ** exp0) * (x1 ** exp1)
        return result
    return evaluate

def load_system(path):
    with open(path, 'r') as f:
        data = json.load(f)
    f1 = build_polynomial(data['polynomials'][0]['terms'])
    f2 = build_polynomial(data['polynomials'][1]['terms'])
    return f1, f2, data['d1'], data['d2'], data['tolerances']

def load_start_system(path):
    with open(path, 'r') as f:
        data = json.load(f)
    g1 = build_polynomial(data['start_polynomials'][0]['terms'])
    g2 = build_polynomial(data['start_polynomials'][1]['terms'])
    gamma = complex(data['gamma_re'], data['gamma_im'])
    return g1, g2, gamma

def compute_jacobian(f1, f2, x):
    eps = 1e-8
    x0 = x[0]
    x1 = x[1]
    J = np.zeros((2, 2), dtype=np.complex128)
    J[0, 0] = (f1(x0 + eps, x1) - f1(x0 - eps, x1)) / (2 * eps)
    J[0, 1] = (f1(x0, x1 + eps) - f1(x0, x1 - eps)) / (2 * eps)
    J[1, 0] = (f2(x0 + eps, x1) - f2(x0 - eps, x1)) / (2 * eps)
    J[1, 1] = (f2(x0, x1 + eps) - f2(x0, x1 - eps)) / (2 * eps)
    return J

def compute_residual(f1, f2, x):
    return np.sqrt(np.abs(f1(x[0], x[1]))**2 + np.abs(f2(x[0], x[1]))**2)

def newton_step(f1, f2, x, t, gamma, tol=1e-10, max_iter=100):
    for _ in range(max_iter):
        H = np.array([gamma * (1 - t) * f1(x[0], x[1]) + t * f1(x[0], x[1]),
                      gamma * (1 - t) * f2(x[0], x[1]) + t * f2(x[0], x[1])], dtype=np.complex128)
        J = compute_jacobian(f1, f2, x)
        try:
            dx = np.linalg.solve(J, -H)
            x += dx
            if np.abs(dx) < tol:
                break
        except np.linalg.LinAlgError:
            return x, False
    return x, True

def euler_predictor(f1, f2, x, t, gamma, dt):
    H = np.array([gamma * (1 - t) * f1(x[0], x[1]) + t * f1(x[0], x[1]),
                  gamma * (1 - t) * f2(x[0], x[1]) + t * f2(x[0], x[1])], dtype=np.complex128)
    dH_dt = np.array([f1(x[0], x[1]), f2(x[0], x[1])], dtype=np.complex128)
    dx_dt = -dH_dt / H
    return x + dx_dt * dt

def path_tracking(f1, f2, g1, g2, gamma, start_x, tol=1e-8, max_steps=1000, dt=1e-3):
    x = start_x.copy()
    t = 0.0
    status = 0
    for _ in range(max_steps):
        if t >= 1.0 - 1e-6:
            x, converged = newton_step(f1, f2, x, t, gamma, tol=1e-10)
            if not converged:
                status = 1
            break
        x_pred = euler_predictor(f1, f2, x, t, gamma, dt)
        x, converged = newton_step(f1, f2, x_pred, t + dt, gamma, tol=1e-8)
        if not converged:
            status = 1
            break
        if np.abs(x[0]) > 1e6 or np.abs(x[1]) > 1e6:
            status = 1
            break
        t += dt
    return x, status

def polish_root(f1, f2, x, tol=1e-10, max_iter=100):
    for _ in range(max_iter):
        J = compute_jacobian(f1, f2, x)
        try:
            dx = np.linalg.solve(J, -np.array([f1(x[0], x[1]), f2(x[0], x[1])], dtype=np.complex128))
            x += dx
            if np.abs(dx) < tol:
                break
        except np.linalg.LinAlgError:
            break
    return x

def main():
    system_path = 'data/system.json'
    start_system_path = 'data/start_system.json'
    start_roots_path = 'data/start_roots.npy'
    
    f1, f2, d1, d2, tolerances = load_system(system_path)
    g1, g2, gamma = load_start_system(start_system_path)
    
    start_roots = np.load(start_roots_path)
    n_paths = d1 * d2
    
    roots = np.zeros((n_paths, 2), dtype=np.complex128)
    residuals = np.zeros(n_paths, dtype=np.float64)
    is_singular = np.zeros(n_paths, dtype=bool)
    is_real = np.zeros(n_paths, dtype=bool)
    condition_numbers = np.zeros(n_paths, dtype=np.float64)
    path_status = np.zeros(n_paths, dtype=np.int8)
    
    for i in range(n_paths):
        start_x = start_roots[i].copy()
        roots[i] = start_x
        path_log = start_x.copy()
        
        x, status = path_tracking(f1, f2, g1, g2, gamma, start_x, max_steps=10000)
        roots[i] = x
        path_status[i] = status
        
        if status == 0:
            x_polished = polish_root(f1, f2, x)
            roots[i] = x_polished
            residual = compute_residual(f1, f2, x_polished)
            residuals[i] = residual
            J = compute_jacobian(f1, f2, x_polished)
            _, sigma_min, _ = np.linalg.svd(J)
            is_singular[i] = sigma_min <= tolerances['singular_sigma_min_tol']
            is_real[i] = max(np.abs(np.imag(x_polished[0])), np.abs(np.imag(x_polished[1]))) <= tolerances['real_tol']
            condition_numbers[i] = np.linalg.cond(J)
        else:
            residuals[i] = 1e10
            is_singular[i] = False
            is_real[i] = False
            condition_numbers[i] = 0.0
    
    np.save('roots.npy', roots)
    np.save('residuals.npy', residuals)
    np.save('is_singular.npy', is_singular)
    np.save('is_real.npy', is_real)
    np.save('condition_numbers.npy', condition_numbers)
    np.save('path_log.npy', start_roots)
    np.save('path_status.npy', path_status)

if __name__ == '__main__':
    main()
```