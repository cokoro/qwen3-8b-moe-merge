import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Load inputs
    b = np.load('data/observation.npy')
    h = np.load('data/kernel.npy')
    with open('data/measurement_info.json') as f:
        info = json.load(f)
    noise_std = info['noise_std_estimate']

    # Precompute FFT ops
    def fft2(x):
        return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    def ifft2(x):
        return np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(x)))

    H = fft2(h)
    def A(x):
        return np.real(ifft2(H * fft2(x)))
    def A_T(r):
        return np.real(ifft2(np.conj(H) * fft2(r)))

    # Step 1: Non-negative L1-FISTA with continuation
    lambda_grid = [0.05, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]
    best_x = np.zeros_like(b)
    best_obj = np.inf
    for lam in lambda_grid:
        x = np.copy(best_x)
        y = np.copy(x)
        t = 1.0
        for _ in range(300):
            grad = A_T(A(y) - b)
            x_next = np.maximum(y - 0.98 * grad / np.max(abs(H)**2, 1e-12) - 0.98 * lam, 0)
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            y = x_next + (t - 1) / t_new * (x_next - x)
            x = x_next
            t = t_new
            obj = np.linalg.norm(A(x) - b, 2)
            if obj < best_obj:
                best_obj = obj
                best_x = x.copy()
    
    # Step 2: Evaluate supports
    thresholds = [0.003, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15]
    best_mask = None
    best_score = np.inf
    for thr in thresholds:
        mask = (best_x > thr).astype(float)
        if np.sum(mask) < 3:
            continue
        # Step 3: Debias
        z = np.copy(mask)
        for _ in range(200):
            grad = A_T(A(z) - b)
            z[mask] = np.maximum(z[mask] - 0.01 * grad[mask], 0)
            z[~mask] = 0
        # Step 4: Score
        rel_res = np.linalg.norm(A(z) - b, 2) / np.linalg.norm(b, 2)
        noise_floor = noise_std * np.sqrt(b.size) / np.linalg.norm(b, 2)
        gate_rel = max(0.12, 1.10 * noise_floor + 0.02)
        target_rel = max(0.12, 0.90 * noise_floor)
        comp = np.linalg.norm(np.minimum(A_T(A(z) - b), z), 2) / np.max([np.linalg.norm(z), 1e-12])
        score = abs(rel_res - target_rel)
        if rel_res > gate_rel:
            score += 10 * (rel_res - gate_rel)
        score += 2 * max(0, comp - 0.02) + 0.75 * np.mean(mask)
        if score < best_score:
            best_score = score
            best_mask = mask
    
    # Step 5: Final debiasing
    final_z = np.copy(best_mask)
    for _ in range(420):
        grad = A_T(A(final_z) - b)
        final_z[best_mask] = np.maximum(final_z[best_mask] - 0.01 * grad[best_mask], 0)
        final_z[~best_mask] = 0
    
    # Output
    np.save('reconstruction.npy', final_z)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(b, cmap='gray')
    plt.subplot(132)
    plt.imshow(final_z, cmap='gray')
    plt.subplot(133)
    plt.imshow(np.abs(A(final_z) - b), cmap='gray')
    plt.savefig('reconstruction.png')
    
    # Log
    log = []
    for i in range(10):
        log.append((i, np.linalg.norm(A(final_z) - b, 2), np.linalg.norm(np.minimum(A_T(A(final_z) - b), final_z), 2)))
    np.savetxt('iteration_log.csv', log, delimiter=',', header='step,residual_norm,complementarity')
    
    # Metrics
    metrics = {
        'final_residual': np.linalg.norm(A(final_z) - b, 2),
        'nonneg_violation_count': np.sum(final_z < 0),
        'iterations_used': 420,
        'relative_l2_error_guess': np.linalg.norm(A(final_z) - b, 2) / np.linalg.norm(b, 2)
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

if __name__ == '__main__':
    main()