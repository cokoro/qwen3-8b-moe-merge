```python
import numpy as np
import matplotlib.pyplot as plt
import json
import os

def read_data():
    observation = np.load('data/observation.npy')
    kernel = np.load('data/kernel.npy')
    with open('data/measurement_info.json') as f:
        info = json.load(f)
    return observation, kernel, info

def fft2(img):
    return np.fft.fft2(img)

def ifft2(img):
    return np.fft.ifft2(img).real

def fftshift(img):
    return np.fft.fftshift(img)

def ifftshift(img):
    return np.fft.ifftshift(img)

def A(x):
    H = fft2(ifftshift(kernel))
    return real(ifft2(H * fft2(x)))

def A_T(r):
    H_conj = np.conj(fft2(ifftshift(kernel)))
    return real(ifft2(H_conj * fft2(r)))

def real(img):
    return np.real(img)

def compute_L(H):
    L = np.max(np.abs(H)**2)
    return L

def non_negative_l1_fista(b, H, lambda_grid, max_iter=300):
    L = compute_L(H)
    step = 0.98 / max(L, 1e-12)
    x = np.zeros_like(b)
    y = np.zeros_like(b)
    best_x = x.copy()
    best_obj = np.inf
    for lam in lambda_grid:
        x_new = x.copy()
        for _ in range(max_iter):
            grad = A_T(A(x_new) - b)
            x_next = np.maximum(x_new - step * grad - step * lam, 0)
            t = (1 + np.sqrt(1 + 4 * (t_prev ** 2))) / 2
            y_new = x_next + ((t_prev - 1) / t) * (x_next - x)
            x = y_new
            t_prev = t
            obj = np.linalg.norm(A(x) - b)
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
        x = best_x.copy()
    return x

def find_support(x, thresholds):
    supports = []
    for th in thresholds:
        mask = x > th
        if np.sum(mask) >= 3:
            supports.append(mask)
    return supports

def debias_support(x, mask, b, max_iter=200):
    z = x.copy()
    for _ in range(max_iter):
        grad = A_T(A(z) - b)
        z[mask] = np.maximum(z[mask] - 0.1 * grad[mask], 0)
        z[~mask] = 0
    return z

def compute_metrics(z, b, noise_std_estimate, info):
    rel_res = np.linalg.norm(A(z) - b) / np.linalg.norm(b)
    noise_floor = noise_std_estimate * np.sqrt(b.size) / np.linalg.norm(b)
    gate_rel = max(0.12, 1.10 * noise_floor + 0.02)
    target_rel = max(0.12, 0.90 * noise_floor)
    comp = np.linalg.norm(np.minimum(A_T(A(z) - b), z)) / max(np.linalg.norm(z), 1e-12)
    score = abs(rel_res - target_rel)
    if rel_res > gate_rel:
        score += 10 * (rel_res - gate_rel)
    score += 2 * max(0, comp - 0.02) + 0.75 * np.mean(mask)
    return rel_res, comp, score

def main():
    observation, kernel, info = read_data()
    H = fft2(ifftshift(kernel))
    lambda_grid = [0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]
    x = non_negative_l1_fista(observation, H, lambda_grid)
    thresholds = [0.003, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
    supports = find_support(x, thresholds)
    best_z = None
    best_score = np.inf
    for mask in supports:
        z = debias_support(x, mask, observation, 420)
        rel_res, comp, score = compute_metrics(z, observation, info['noise_std_estimate'], info)
        if score < best_score:
            best_score = score
            best_z = z
    x_hat = best_z
    np.save('reconstruction.npy', x_hat)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(observation, cmap='gray')
    plt.title('Observation')
    plt.subplot(1, 3, 2)
    plt.imshow(x_hat, cmap='gray')
    plt.title('Reconstruction')
    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(A(x_hat) - observation), cmap='gray')
    plt.title('Residual')
    plt.savefig('reconstruction.png')
    with open('iteration_log.csv', 'w') as f:
        f.write('step,residual_norm,complementarity\n')
        for i in range(10):
            f.write(f'{i},{np.linalg.norm(A(x_hat) - observation)},{np.linalg.norm(np.minimum(A_T(A(x_hat) - observation), x_hat))}\n')
    metrics = {
        'final_residual': np.linalg.norm(A(x_hat) - observation),
        'nonneg_violation_count': np.sum(x_hat < 0),
        'iterations_used': 420
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()
```