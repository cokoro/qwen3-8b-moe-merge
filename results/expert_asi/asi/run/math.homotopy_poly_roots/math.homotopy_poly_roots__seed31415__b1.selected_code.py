import numpy as np
import scipy.optimize as opt
import scipy.linalg as la

def read_polynomial(path, name):
    terms = []
    for term in path['polynomials'][name]['terms']:
        coeff = complex(term['coeff_re'], term['coeff_im'])
        exponents = tuple(term['exponents'])
        terms.append((coeff, exponents))
    return terms

def evaluate_polynomial(terms, x):
    res = 0 + 0j
    for coeff, exponents in terms:
        res += coeff * x[0]**exponents[0] * x[1]**exponents[1]
    return res

def build_jacobian(terms, x):
    J = np.zeros((2, 2), dtype=np.complex128)
    for idx, (coeff, (a, b)) in enumerate(terms):
        dxdx = coeff * a * x[0]**(a-1) * x[1]**b
        dxdy = coeff * b * x[0]**a * x[1]**(b-1)
        J[0, 0] += dxdx
        J[0, 1] += dxdy
        J[1, 0] += dxdx
        J[1, 1] += dxdy
    return J

def homotopy_step(F_terms, G_terms, gamma, t, x, delta_t):
    H = gamma * (1 - t) * evaluate_polynomial(G_terms, x) + t * evaluate_polynomial(F_terms, x)
    J = build_jacobian(F_terms + G_terms, x)
    dxdt = -np.conj(H) / np.conj(J[0, 0]) if np.abs(J[0, 0]) > 1e-15 else -np.conj(H) / np.conj(J[1, 1])
    return x + delta_t * dxdt

def solve_path(F_terms, G_terms, gamma, start_x, tol=1e-8, max_steps=1000):
    x = np.copy(start_x)
    t = 0.0
    steps = 0
    log = [x.copy()]
    for _ in range(max_steps):
        delta_t = 1e-4
        x_pred = homotopy_step(F_terms, G_terms, gamma, t, x, delta_t)
        J = build_jacobian(F_terms + G_terms, x_pred)
        try:
            dx = np.linalg.solve(J, -evaluate_polynomial(F_terms + G_terms, x_pred))
            x = x_pred + dx
            t += delta_t
            log.append(x.copy())
            if np.abs(evaluate_polynomial(F_terms, x)) < tol:
                break
        except np.linalg.LinAlgError:
            delta_t *= 0.1
    return x, log

def main():
    import json
    with open('data/system.json') as f:
        system = json.load(f)
    with open('data/start_system.json') as f:
        start = json.load(f)
    start_roots = np.load('data/start_roots.npy')
    
    F_terms = read_polynomial(system, 'f1') + read_polynomial(system, 'f2')
    G_terms = read_polynomial(start, 'g1') + read_polynomial(start, 'g2')
    gamma = complex(start['gamma_re'], start['gamma_im'])
    
    n_paths = system['n_paths']
    roots = np.zeros((n_paths, 2), dtype=np.complex128)
    residuals = np.zeros(n_paths, dtype=np.float64)
    is_singular = np.zeros(n_paths, dtype=bool)
    is_real = np.zeros(n_paths, dtype=bool)
    condition_numbers = np.zeros(n_paths, dtype=np.float64)
    path_status = np.zeros(n_paths, dtype=np.int8)
    
    for i in range(n_paths):
        x = start_roots[i]
        x_final, log = solve_path(F_terms, G_terms, gamma, x)
        roots[i] = x_final
        f = evaluate_polynomial(F_terms, x_final)
        residuals[i] = np.linalg.norm(f)
        if np.isnan(residuals[i]):
            path_status[i] = 1
            residuals[i] = 1e10
            roots[i] = 1e8 + 1e8j
        else:
            path_status[i] = 0
            J = build_jacobian(F_terms, x_final)
            _, s, _ = np.linalg.svd(J)
            is_singular[i] = s.min() < system['tolerances']['singular_sigma_min_tol']
            is_real[i] = max(abs(np.imag(x_final[0])), abs(np.imag(x_final[1]))) <= system['tolerances']['real_tol']
            condition_numbers[i] = s.max() / s.min()
    
    np.save('roots.npy', roots)
    np.save('residuals.npy', residuals)
    np.save('is_singular.npy', is_singular)
    np.save('is_real.npy', is_real)
    np.save('condition_numbers.npy', condition_numbers)
    np.save('path_status.npy', path_status)
    np.save('path_log.npy', start_roots)