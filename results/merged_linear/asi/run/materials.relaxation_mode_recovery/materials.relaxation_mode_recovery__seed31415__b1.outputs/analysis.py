import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint
import time

# Load the data
data = pd.read_csv('data/curves.csv', index_col=0)
M = len(data)
N = data.shape[1]
X = data.values

# Preprocess the data
s = np.zeros(M)
for m in range(M):
    s[m] = np.max([np.ptp(np.abs(X[m, :])), np.std(X[m, :][::2]), np.std(X[m, :][1::2]), 1e-10])

# Define the objective function
def objective(p, X, s, M, N):
    ell = p[:M]
    phi = p[M:M+M]
    a = p[M+M:]
    y = np.zeros((M, N), dtype=np.complex128)
    for m in range(M):
        for n in range(N):
            real = 0.0
            imag = 0.0
            for k in range(M):
                t = np.exp(ell[k])
                denom = 1.0 + 1j * X[n] * t ** phi[k]
                real += a[k] / denom.real
                imag += a[k] / denom.imag
            y[m, n] = real + 1j * imag
    residual = (X - y) / s
    return np.sum(np.abs(residual) ** 2)

# Define the bounds
bounds = Bounds(
    lb=-np.log10(X.max()) + 0.20,
    ub=-np.log10(X.min()) - 0.20,
    keep_feasible=True
)

# Define the constraints
def constraint1(p, X, s, M, N):
    ell = p[:M]
    phi = p[M:M+M]
    a = p[M+M:]
    return np.sum((ell - ell[0]) ** 2)

constraint1 = NonlinearConstraint(constraint1, 0.0, 1e-8)

def constraint2(p, X, s, M, N):
    ell = p[:M]
    phi = p[M:M+M]
    a = p[M+M:]
    return np.sum((phi - 0.78) ** 2)

constraint2 = NonlinearConstraint(constraint2, 0.0, 1e-8)

def constraint3(p, X, s, M, N):
    ell = p[:M]
    phi = p[M:M+M]
    a = p[M+M:]
    return np.sum((a - a[0]) ** 2)

constraint3 = NonlinearConstraint(constraint3, 0.0, 1e-8)

# Generate initial guesses
np.random.seed(0)
initial_guesses = []
for m in range(M):
    ell = np.random.uniform(-np.log10(X.max()) + 0.20, -np.log10(X.min()) - 0.20, size=M)
    phi = np.random.uniform(0.78, 0.94, size=M)
    a = np.random.uniform(0.1, 1.0, size=M)
    initial_guesses.append(np.concatenate([ell, phi, a]))

# Run the optimization
results = []
for i, guess in enumerate(initial_guesses):
    print(f'Running optimization {i + 1}/{len(initial_guesses)}')
    result = minimize(
        objective,
        guess,
        args=(X, s, M, N),
        method='SLSQP',
        bounds=bounds,
        constraints=[constraint1, constraint2, constraint3],
        options={'disp': False, 'ftol': 1e-12}
    )
    results.append(result)

# Evaluate the results
best_result = min(results, key=lambda x: x.fun)
best_ell = best_result.x[:M]
best_phi = best_result.x[M:M+M]
best_a = best_result.x[M+M:]

# Print the results
print(f'Best objective value: {best_result.fun:.4f}')
print(f'Best ell: {best_ell}')
print(f'Best phi: {best_phi}')
print(f'Best a: {best_a}')

# Save the results
np.savetxt('results/modes.csv', np.column_stack([np.exp(best_ell), best_a]), delimiter=',')