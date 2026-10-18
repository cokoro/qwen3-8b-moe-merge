```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
import scipy.io
import json

# Load data
b = np.load('data/observation.npy')
h = np.load('data/kernel.npy')
with open('data/measurement_info.json') as f:
    info = json.load(f)
noise_std = float(info['noise_std_estimate'])
image_shape = (128, 128)

def apply_operator(x):
    """Apply A(x) = real(ifft2(H * fft2(x)))"""
    H = fft.fft2(fft.ifftshift(h))
    return np.real(fft.ifft2(H * fft.fft2(x)))

def adjoint_operator(r):
    """Apply A^T(r) = real(ifft2(conj(H) * fft2(r)))"""
    H = fft.fft2(fft.ifftshift(h))
    return np.real(fft.ifft2(np.conj(H) * fft.fft2(r)))

def compute_residual(x):
    """Compute ||A(x) - b||_2"""
    return np.sqrt(np.sum((apply_operator(x) - b)**2))

def compute_complementarity(x):
    """Compute ||min(A^T(A(x)-b), x)||_2 / ||x||_2"""
    a = adjoint_operator(apply_operator(x) - b)
    comp = np.sqrt(np.sum(np.minimum(a, x)**2)) / np.max([np.linalg.norm(x), 1e-12])
    return comp

def debias_support(x, mask, steps=200):
    """Debias a support with projected LS iterations"""
    x = x.copy()
    for _ in range(steps):
        grad = adjoint_operator(apply_operator(x) - b)
        x[mask] = np.maximum(x[mask] - 0.05 * grad[mask], 0)
        x[~mask] = 0
    return x

def l1_fista(x0, lambda_grid, max_iter=300):
    """Non-negative L1-FISTA with continuation"""
    x = x0.copy()
    y = x.copy()
    t = 1.0
    history = []
    for lam in lambda_grid:
        for _ in range(max_iter):
            grad = adjoint_operator(apply_operator(y) - b)
            step = 0.98 / max(np.max(np.abs(apply_operator(y) - b)**2), 1e-12)
            x_next = np.maximum(y - step * grad - step * lam, 0)
            t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
            y = x_next + ((t - 1) / t_next) * (x_next - x)
            x = x_next
            t = t_next
            history.append((compute_residual(x), compute_complementarity(x)))
    return x, history

def select_support(x, mask, steps=420):
    """Select the best support using diagnostics"""
    x_best = x.copy()
    x_masked = x * mask
    for _ in range(steps):
        grad = adjoint_operator(apply_operator(x_masked) - b)
        x_masked[mask] = np.maximum(x_masked[mask] - 0.05 * grad[mask], 0)
        x_masked[~mask] = 0
    return x_masked

# Initialize with zeros
x0 = np.zeros(image_shape)
x, history = l1_fista(x0, [0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005], 300)

# Select the best support
x_hat = select_support(x, np.abs(x) > 0.005)
x_hat = np.maximum(x_hat, 0)

# Save output
np.save('reconstruction.npy', x_hat)
plt.imsave('reconstruction.png', np.vstack([b, x_hat, np.abs(apply_operator(x_hat) - b)]), cmap='gray')
np.savetxt('iteration_log.csv', history, header='step,residual_norm,complementarity')
```