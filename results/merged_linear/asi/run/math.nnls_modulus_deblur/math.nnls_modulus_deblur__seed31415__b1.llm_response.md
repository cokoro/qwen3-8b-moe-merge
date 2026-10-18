```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
b = np.load('data/observation.npy')
h = np.load('data/kernel.npy')
with open('data/measurement_info.json') as f:
    info = json.load(f)
noise_std = info['noise_std_estimate']
image_shape = info['image_shape']

# FFT operations
def fft2(x):
    return np.fft.fftshift(np.fft.fft2(x))

def ifft2(x):
    return np.fft.ifftshift(np.fft.ifft2(x))

def apply_operator(x):
    return real(ifft2(fft2(x)))

def apply_adjoint(r):
    return real(ifft2(conj(fft2(r))))

# L1-FISTA with continuation
def non_negative_l1_fista(lambda_grid, max_iter=300):
    x = np.zeros_like(b)
    y = x.copy()
    t = 1.0
    best_x = x.copy()
    best_obj = np.inf
    for lam in lambda_grid:
        x = y.copy()
        for _ in range(max_iter):
            grad = apply_adjoint(apply_operator(x) - b)
            x_next = np.maximum(x - step * grad - step * lam, 0)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = x_next + ((t - 1) / t_next) * (x_next - x)
            t = t_next
        obj = np.linalg.norm(apply_operator(x) - b)
        if obj < best_obj:
            best_obj = obj
            best_x = x.copy()
    return best_x

# Thresholding and debiasing
def find_best_support(x, thresholds):
    best_mask = None
    best_score = np.inf
    for thresh in thresholds:
        mask = x > thresh * np.max(x)
        if np.sum(mask) < 3:
            continue
        z = np.zeros_like(x)
        z[mask] = x[mask]
        for _ in range(200):
            grad = apply_adjoint(apply_operator(z) - b)
            z[mask] = np.maximum(z[mask] - step * grad[mask], 0)
        obj = np.linalg.norm(apply_operator(z) - b)
        rel_res = obj / np.linalg.norm(b)
        noise_floor = noise_std * np.sqrt(b.size) / np.linalg.norm(b)
        gate_rel = max(0.12, 1.10 * noise_floor + 0.02)
        target_rel = max(0.12, 0.90 * noise_floor)
        comp = np.linalg.norm(np.minimum(apply_adjoint(apply_operator(z) - b), z)) / max(np.linalg.norm(z), 1e-12)
        score = abs(rel_res - target_rel)
        if rel_res > gate_rel:
            score += 10 * (rel_res - gate_rel)
        score += 2 * max(0, comp - 0.02) + 0.75 * np.mean(mask)
        if score < best_score:
            best_score = score
            best_mask = mask
    return best_mask

# Main execution
lambda_grid = [0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]
max_iter = 300
step = 0.98 / max(np.max(np.abs(fft2(h)**2)), 1e-12)
x = non_negative_l1_fista(lambda_grid, max_iter)
thresholds = [0.003, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
best_mask = find_best_support(x, thresholds)

# Debias selected support
z = np.zeros_like(b)
z[best_mask] = x[best_mask]
for _ in range(420):
    grad = apply_adjoint(apply_operator(z) - b)
    z[best_mask] = np.maximum(z[best_mask] - step * grad[best_mask], 0)

# Save outputs
np.save('reconstruction.npy', z)
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.imshow(b, cmap='gray')
plt.subplot(132)
plt.imshow(z, cmap='gray')
plt.subplot(133)
plt.imshow(np.abs(apply_operator(z) - b), cmap='gray')
plt.savefig('reconstruction.png')

# Log iteration metrics
iteration_log = []
for i in range(10):
    iteration_log.append([i, np.linalg.norm(apply_operator(z) - b), np.linalg.norm(np.minimum(apply_adjoint(apply_operator(z) - b), z)) / max(np.linalg.norm(z), 1e-12)])
np.savetxt('iteration_log.csv', iteration_log, delimiter=',')

# Save metrics
metrics = {
    'final_residual': np.linalg.norm(apply_operator(z) - b),
    'nonneg_violation_count': np.sum(z < 0),
    'iterations_used': 420,
    'relative_l2_error_guess': np.linalg.norm(apply_operator(z) - b) / np.linalg.norm(b)
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
```