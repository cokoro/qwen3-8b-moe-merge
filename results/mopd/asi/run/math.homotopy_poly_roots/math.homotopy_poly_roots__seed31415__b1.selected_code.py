import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
import json
import os

# Load data
with open('data/system.json') as f:
    system = json.load(f)
with open('data/start_system.json') as f:
    start_system = json.load(f)
start_roots = np.load('data/start_roots.npy')

# Build F and G
def build_polynomial(terms):
    def poly(x):
        res = np.zeros_like(x[0])
        for term in terms:
            res += term['coeff_re'] * x[0]**term['exponents'][0] * x[1]**term['exponents'][1] + 1j * term['coeff_im'] * x[0]**term['exponents'][0] * x[1]**term['exponents'][1]
        return res
    return poly

F = build_polynomial(system['polynomials'][0]['terms'])
G = build_polynomial(start_system['start_polynomials'][0]['terms'])
G2 = build_polynomial(start_system['start_polynomials'][1]['terms'])

# Build H(x, t) = gamma*(1-t)*G(x) + t*F(x)
gamma = complex(start_system['gamma_re'], start_system['gamma_im'])

def H(x, t):
    return gamma * (1 - t) * G(x) + t * F(x)

def dH_dx(x, t):
    x0, x1 = x
    res = np.zeros((2, 2), dtype=np.complex128)
    for term in start_system['start_polynomials'][0]['terms']:
        res[0, 0] += term['coeff_re'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1] + 1j * term['coeff_im'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1]
        res[0, 1] += term['coeff_re'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1) + 1j * term['coeff_im'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1)
    for term in system['polynomials'][0]['terms']:
        res[0, 0] += term['coeff_re'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1] + 1j * term['coeff_im'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1]
        res[0, 1] += term['coeff_re'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1) + 1j * term['coeff_im'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1)
    for term in start_system['start_polynomials'][1]['terms']:
        res[1, 0] += term['coeff_re'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1] + 1j * term['coeff_im'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1]
        res[1, 1] += term['coeff_re'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1) + 1j * term['coeff_im'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1)
    for term in system['polynomials'][1]['terms']:
        res[1, 0] += term['coeff_re'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1] + 1j * term['coeff_im'] * term['exponents'][0] * x0**(term['exponents'][0] - 1) * x1**term['exponents'][1]
        res[1, 1] += term['coeff_re'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1) + 1j * term['coeff_im'] * term['exponents'][1] * x0**term['exponents'][0] * x1**(term['exponents'][1] - 1)
    return res

def dH_dt(x, t):
    return gamma * (-G(x) + F(x))

# Path tracking
n_paths = system['n_paths']
roots = np.zeros((n_paths, 2), dtype=np.complex128)
residuals = np.zeros(n_paths, dtype=np.float64)
is_singular = np.zeros(n_paths, dtype=bool)
is_real = np.zeros(n_paths, dtype=bool)
condition_numbers = np.zeros(n_paths, dtype=np.float64)
path_status = np.zeros(n_paths, dtype=np.int8)
path_log = start_roots.copy()

for i in range(n_paths):
    x = start_roots[i].copy()
    t = 0.0
    for _ in range(200): # Max steps
        # Predictor
        dt = 0.1
        dx_pred = -dH_dt(x, t) / dH_dx(x, t)
        x_pred = x + dt * dx_pred
        t_pred = t + dt
        
        # Corrector
        def H_func(x):
            return H(x, t_pred)
        def J_func(x):
            return dH_dx(x, t_pred)
        
        # Newton-Raphson
        x_new = opt.root(lambda x: H_func(x), x_pred, jac=J_func, method='hybr').x
        x = x_new
        t = t_pred
        
        # Check convergence
        if np.abs(x[0]) > 1e6 or np.abs(x[1]) > 1e6:
            path_status[i] = 1
            roots[i] = 1e8 + 1e8j
            break
        elif np.linalg.norm(H(x, t)) < system['tolerances']['residual_pass_tol']:
            # Polishing
            def F_func(x):
                return F(x)
            def J_func(x):
                return np.array([[term['coeff_re'] * term['exponents'][0] * x[0]**(term['exponents'][0]-1) * x[1]**term['exponents'][1] + 1j * term['coeff_im'] * term['exponents'][0] * x[0]**(term['exponents'][0]-1) * x[1]**term['exponents'][1] for term in system['polynomials'][0]['terms']],
                            [term['coeff_re'] * term['exponents'][1] * x[0]**term['exponents'][0] * x[1]**(term['exponents'][1]-1) + 1j * term['coeff_im'] * term['exponents'][1] * x[0]**term['exponents'][0] * x[1]**(term['exponents'][1]-1) for term in system['polynomials'][1]['terms']]])
            x_new = opt.root(lambda x: F_func(x), x, jac=J_func, method='hybr').x
            x = x_new
            residuals[i] = np.linalg.norm(F(x))
            J = J_func(x)
            U, s, Vh = np.linalg.svd(J)
            is_singular[i] = s[-1] < system['tolerances']['singular_sigma_min_tol']
            is_real[i] = max(abs(x[0].imag), abs(x[1].imag)) < system['tolerances']['real_tol']
            condition_numbers[i] = s[0] / s[-1]
            path_status[i] = 0
            roots[i] = x
            break
        else:
            # Continue tracking
            pass
    else:
        path_status[i] = 1
        roots[i] = 1e8 + 1e8j

np.save('roots.npy', roots)
np.save('residuals.npy', residuals)
np.save('is_singular.npy', is_singular)
np.save('is_real.npy', is_real)
np.save('condition_numbers.npy', condition_numbers)
np.save('path_log.npy', path_log)
np.save('path_status.npy', path_status)