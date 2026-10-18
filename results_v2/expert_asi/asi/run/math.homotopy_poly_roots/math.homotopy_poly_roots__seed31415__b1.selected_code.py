import numpy as np
import scipy.linalg
import json

def parse_polynomial(json_poly):
    terms = []
    for term in json_poly['terms']:
        coeff_re = term['coeff_re']
        coeff_im = term['coeff_im']
        exponents = term['exponents']
        terms.append((coeff_re + 1j * coeff_im, exponents))
    return terms

def evaluate_poly(poly_terms, x0, x1):
    res = 0.0
    for coeff, (a, b) in poly_terms:
        res += coeff * (x0**a) * (x1**b)
    return res

def jacobian(poly_terms, x0, x1):
    jac = np.zeros((len(poly_terms), 2), dtype=np.complex128)
    for i, (coeff, (a, b)) in enumerate(poly_terms):
        jac[i, 0] = coeff * a * (x0**(a-1)) * (x1**b)
        jac[i, 1] = coeff * b * (x0**a) * (x1**(b-1))
    return jac

def solve_homotopy(data_path='data/system.json', start_path='data/start_system.json', roots_path='data/start_roots.npy'):
    with open(data_path) as f:
        system = json.load(f)
    with open(start_path) as f:
        start_system = json.load(f)
    
    # Load start points
    start_roots = np.load(roots_path)
    n_paths = len(start_roots)
    
    # Parse polynomials
    f_terms = parse_polynomial(system['polynomials'][0])
    g_terms = parse_polynomial(start_system['start_polynomials'][0])
    g2_terms = parse_polynomial(start_system['start_polynomials'][1])
    
    # Initialize log
    path_log = start_roots.copy()
    roots = np.zeros((n_paths, 2), dtype=np.complex128)
    residuals = np.zeros(n_paths, dtype=np.float64)
    is_singular = np.zeros(n_paths, dtype=bool)
    is_real = np.zeros(n_paths, dtype=bool)
    condition_numbers = np.zeros(n_paths, dtype=np.float64)
    path_status = np.zeros(n_paths, dtype=np.int8)
    
    # Define homotopy
    gamma = complex(start_system['gamma_re'], start_system['gamma_im'])
    t = np.linspace(0.0, 1.0, 1000)
    
    # Track each path
    for i in range(n_paths):
        x = start_roots[i]
        x0, x1 = x[0], x[1]
        for step in range(len(t)):
            t_val = t[step]
            # Evaluate H(x, t_val)
            H = gamma * (1 - t_val) * evaluate_poly(g_terms, x0, x1) + t_val * evaluate_poly(f_terms, x0, x1)
            # Newton step
            J = jacobian(f_terms, x0, x1)
            dx = -np.linalg.solve(J, H)
            x += dx
            # Check convergence
            if np.abs(H) < 1e-12:
                roots[i] = x
                residuals[i] = np.abs(H)
                J = jacobian(f_terms, x0, x1)
                u, s, vh = np.linalg.svd(J)
                is_singular[i] = s[-1] < system['tolerances']['singular_sigma_min_tol']
                is_real[i] = max(np.imag(x0), np.imag(x1)) < system['tolerances']['real_tol']
                condition_numbers[i] = s[0] / s[-1]
                path_status[i] = 0
            else:
                path_status[i] = 1
                roots[i] = np.inf + np.inf * 1j
                residuals[i] = 1e10
                is_singular[i] = False
                is_real[i] = False
                condition_numbers[i] = 0.0
    
    np.save('roots.npy', roots)
    np.save('residuals.npy', residuals)
    np.save('is_singular.npy', is_singular)
    np.save('is_real.npy', is_real)
    np.save('condition_numbers.npy', condition_numbers)
    np.save('path_status.npy', path_status)
    np.save('path_log.npy', path_log)