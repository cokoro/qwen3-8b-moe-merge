```py
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d
from scipy.linalg import norm

# Load inputs
observation = np.load('data/observation.npy')
kernel = np.load('data/kernel.npy')
with open('data/measurement_info.json') as f:
    info = json.load(f)
image_shape = info['image_shape']
noise_std_estimate = info['noise_std_estimate']

# Forward model
def A(x):
    H = fft2(ifftshift(kernel))
    return real(ifft2(H * fft2(x)))

def A_T(r):
    H = fft2(ifftshift(kernel))
    return real(ifft2(conj(H) * fft2(r)))

# L1-FISTA with continuation
def solve_l1_fista(b, H, lambda_grid, max_iter=300):
    L = np.max(np.abs(H)**2)
    step = 0.98 / max(L, 1e-12)
    x = np.zeros_like(b)
    y = np.zeros_like(b)
    best_x = np.zeros_like(b)
    best_obj = np.inf
    for lambda_val in lambda_grid:
        x = np.zeros_like(b)
        y = np.zeros_like(b)
        for i in range(max_iter):
            grad = A_T(A(y) - b)
            x_next = np.maximum(y - step * grad - step * lambda_val, 0)
            t = (1 + np.sqrt(1 + 4 * (i+1)**2)) / 2
            y = x_next + ((t - 1) / t) * (x_next - x)
            x = x_next
        obj = norm(A(x) - b)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
    return best_x

# Thresholding
def threshold(x, threshold):
    return np.maximum(x - threshold, 0)

# Debiasing
def debias(x, b, kernel, max_iter=200):
    H = fft2(ifftshift(kernel))
    x_debiased = x.copy()
    for i in range(max_iter):
        grad = A_T(A(x_debiased) - b)
        x_debiased = np.maximum(x_debiased - step * grad, 0)
    return x_debiased

# Selecting support
def select_support(x, b, kernel, noise_std_estimate):
    L = np.max(np.abs(H)**2)
    step = 0.98 / max(L, 1e-12)
    x = np.zeros_like(b)
    y = np.zeros_like(b)
    best_x = np.zeros_like(b)
    best_obj = np.inf
    for lambda_val in lambda_grid:
        x = np.zeros_like(b)
        y = np.zeros_like(b)
        for i in range(max_iter):
            grad = A_T(A(y) - b)
            x_next = np.maximum(y - step * grad - step * lambda_val, 0)
            t = (1 + np.sqrt(1 + 4 * (i+1)**2)) / 2
            y = x_next + ((t - 1) / t) * (x_next - x)
            x = x_next
        obj = norm(A(x) - b)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
    return best_x

# Final reconstruction
def final_reconstruction(b, kernel, noise_std_estimate):
    H = fft2(ifftshift(kernel))
    x = np.zeros_like(b)
    y = np.zeros_like(b)
    best_x = np.zeros_like(b)
    best_obj = np.inf
    for lambda_val in lambda_grid:
        x = np.zeros_like(b)
        y = np.zeros_like(b)
        for i in range(max_iter):
            grad = A_T(A(y) - b)
            x_next = np.maximum(y - step * grad - step * lambda_val, 0)
            t = (1 + np.sqrt(1 + 4 * (i+1)**2)) / 2
            y = x_next + ((t - 1) / t) * (x_next - x)
            x = x_next
        obj = norm(A(x) - b)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
    return best_x

# Final output
x_hat = final_reconstruction(observation, kernel, noise_std_estimate)
np.save('reconstruction.npy', x_hat)
plt.figure(figsize=(10, 10))
plt.subplot(131)
plt.imshow(observation, cmap='gray')
plt.subplot(132)
plt.imshow(x_hat, cmap='gray')
plt.subplot(133)
plt.imshow(np.abs(A(x_hat) - observation), cmap='gray')
plt.savefig('reconstruction.png')

# Log
iteration_log = []
x = np.zeros_like(observation)
y = np.zeros_like(observation)
for i in range(10):
    grad = A_T(A(y) - observation)
    x_next = np.maximum(y - step * grad - step * lambda_val, 0)
    t = (1 + np.sqrt(1 + 4 * (i+1)**2)) / 2
    y = x_next + ((t - 1) / t) * (x_next - x)
    x = x_next
    iteration_log.append([i+1, norm(A(x) - observation), norm(A_T(A(x) - observation))])
np.savetxt('iteration_log.csv', iteration_log, delimiter=',')

# Metrics
metrics = {}
metrics['final_residual'] = norm(A(x) - observation)
metrics['nonneg_violation_count'] = np.sum(x < 0)
metrics['iterations_used'] = 10
metrics['relative_l2_error_guess'] = norm(A(x) - observation) / norm(observation)
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
```