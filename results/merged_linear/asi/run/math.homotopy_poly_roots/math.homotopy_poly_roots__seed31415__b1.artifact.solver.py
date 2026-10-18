import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import json
import os

# Load system data
with open('data/system.json') as f:
    system = json.load(f)
d1, d2 = system['d1'], system['d2']
n_paths = d1 * d2
tolerances = system['tolerances']
polynomials = system['polynomials']

# Load start system data
with open('data/start_system.json') as f:
    start_system = json.load(f)
gamma = complex(start_system['gamma_re'], start_system['gamma_im'])
start_polynomials = start_system['start_polynomials']

# Load start points
path_log = np.load('data/start_roots.npy')

# Define polynomial evaluation
def eval_poly(p, x):
    x0, x1 = x
    res = 0
    for term in p['terms']:
        coeff = complex(term['coeff_re'], term['coeff_im'])
        exponents = term['exponents']
        res += coeff * (x0 ** exponents[0]) * (x1 ** exponents[1])
    return res

# Build start system
start_f1 = eval_poly(start_polynomials[0], path_log[0])
start_f2 = eval_poly(start_polynomials[1], path_log[0])

# Define homotopy
def H(x, t):
    f1 = eval_poly(polynomials[0], x)
    f2 = eval_poly(polynomials[1], x)
    return gamma * (1 - t) * start_f1 + t * f1, gamma * (1 - t) * start_f2 + t * f2

# Define Jacobian
def J(x):
    x0, x1 = x
    J11 = 0
    J12 = 0
    J21 = 0
    J22 = 0
    for term in polynomials[0]['terms']:
        coeff = complex(term['coeff_re'], term['coeff_im'])
        exponents = term['exponents']
        J11 += coeff * exponents[0] * (x0 ** (exponents[0] - 1)) * (x1 ** exponents[1])
        J12 += coeff * exponents[1] * (x0 ** exponents[0]) * (x1 ** (exponents[1] - 1))
    for term in polynomials[1]['terms']:
        coeff = complex(term['coeff_re'], term['coeff_im'])
        exponents = term['exponents']
        J21 += coeff * exponents[0] * (x0 ** (exponents[0] - 1)) * (x1 ** exponents[1])
        J22 += coeff * exponents[1] * (x0 ** exponents[0]) * (x1 ** (exponents[1] - 1))
    return np.array([[J11, J12], [J21, J22]])

# Predictor-corrector path tracking
roots = np.zeros((n_paths, 2), dtype=np.complex128)
residuals = np.zeros(n_paths, dtype=np.float64)
is_singular = np.zeros(n_paths, dtype=bool)
is_real = np.zeros(n_paths, dtype=bool)
condition_numbers = np.zeros(n_paths, dtype=np.float64)
path_status = np.zeros(n_paths, dtype=np.int8)

for i in range(n_paths):
    x = path_log[i].copy()
    t = 0.0
    dt = 0.01
    converged = False
    for _ in range(1000):
        if converged:
            break
        # Predict
        f1, f2 = H(x, t)
        J = J(x)
        df1dt = -gamma * start_f1
        df2dt = -gamma * start_f2
        dxdt = -la.solve(J, np.array([df1dt, df2dt]))
        x += dxdt * dt
        t += dt
        # Correct
        f1, f2 = H(x, t)
        J = J(x)
        residual = np.abs(f1) + np.abs(f2)
        if residual < tolerances['residual_pass_tol']:
            converged = True
        else:
            dx = la.solve(J, -np.array([f1, f2]))
            x += dx
            if np.abs(dx) < tolerances['match_tol']:
                converged = True
        if np.abs(x[0]) > 1e6 or np.abs(x[1]) > 1e6:
            path_status[i] = 1
            roots[i] = 1e8 + 1e8j
            residuals[i] = 1e10
            is_singular[i] = False
            is_real[i] = False
            condition_numbers[i] = 0
            break
    if not converged:
        path_status[i] = 1
        roots[i] = 1e8 + 1e8j
        residuals[i] = 1e10
        is_singular[i] = False
        is_real[i] = False
        condition_numbers[i] = 0
    else:
        # Post-processing
        f1, f2 = H(x, 1.0)
        J = J(x)
        residuals[i] = np.abs(f1) + np.abs(f2)
        U, s, Vh = np.linalg.svd(J)
        is_singular[i] = s[-1] <= tolerances['singular_sigma_min_tol']
        is_real[i] = max(np.abs(np.imag(x[0])), np.abs(np.imag(x[1]))) <= tolerances['real_tol']
        condition_numbers[i] = s[0] / s[-1] if s[-1] != 0 else 0
        roots[i] = x

# Save outputs
np.save('roots.npy', roots)
np.save('residuals.npy', residuals)
np.save('is_singular.npy', is_singular)
np.save('is_real.npy', is_real)
np.save('condition_numbers.npy', condition_numbers)
np.save('path_log.npy', path_log)
np.save('path_status.npy', path_status)