```bash
cat > solution.py << 'EOF'
import numpy as np

# Load inputs
h = np.load('data/kernel.npy')
b = np.load('data/observation.npy')
info = np.load('data/measurement_info.json', allow_pickle=True).item()
noise_std_estimate = info['noise_std_estimate']
image_shape = tuple(info['image_shape'])

# Step 1: Create forward operator
def A(x):
    h_shifted = np.fft.ifftshift(h)
    H = np.fft.fft2(h_shifted)
    y = np.fft.fft2(x)
    z = H * y
    return np.fft.irfft2(z).real

def AT(r):
    h_shifted = np.fft.ifftshift(h)
    H = np.fft.fft2(h_shifted)
    y = np.fft.fft2(r)
    z = np.conj(H) * y
    return np.fft.irfft2(z).real

# Step 2: Solve non-negative L1-FISTA with continuation
def solve_non_negative_l1_FISTA(A, AT, b, x0, lambda_grid, max_iter=300):
    # Step 2.1: Initialize
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    best_objective = None
    best_x = x0.copy()

    # Step 2.2: Iterate for each lambda in lambda_grid
    for lambda_val in lambda_grid:
        # Step 2.2.1: Initialize for this lambda
        x = best_x.copy()
        y = x.copy()
        t = 1.0

        # Step 2.2.2: Iterate for this lambda
        for step in range(max_iter):
            # Step 2.2.2.1: Compute gradient
            grad = AT(A(x) - b)

            # Step 2.2.2.2: Update x
            x_next = np.maximum(y - 0.98 * grad - 0.98 * lambda_val, 0.0)

            # Step 2.2.2.3: Update t
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2

            # Step 2.2.2.4: Update y
            y = x_next + ((t - 1) / t_next) * (x_next - x)

            # Step 2.2.2.5: Update x
            x = x_next.copy()

            # Step 2.2.2.6: Track best objective
            objective = 0.5 * np.linalg.norm(A(x) - b)**2 + lambda_val * np.sum(x)
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_x = x.copy()

        # Step 2.2.3: Store best_x for this lambda
        best_x = x.copy()

    return best_x

# Step 3: Find best support
def find_best_support(A, AT, b, x0, lambda_grid, threshold_grid, max_iter=200):
    # Step 3.1: Initialize
    best_score = None
    best_mask = None

    # Step 3.2: Iterate over threshold_grid
    for threshold in threshold_grid:
        # Step 3.2.1: Threshold to get mask
        mask = x0 > threshold

        # Step 3.2.2: Skip if too sparse
        if np.sum(mask) < 3:
            continue

        # Step 3.2.3: Debias with projected least-squares
        x = x0.copy()
        for step in range(max_iter):
            grad = AT(A(x) - b)
            x[mask] = np.maximum(x[mask] - 0.98 * grad[mask], 0.0)

        # Step 3.2.4: Track best score
        score = evaluate_support(A, AT, b, x, mask, noise_std_estimate)
        if best_score is None or score < best_score:
            best_score = score
            best_mask = mask

    return best_mask

# Step 4: Evaluate support
def evaluate_support(A, AT, b, x, mask, noise_std_estimate):
    # Step 4.1: Compute residual norm
    residual = A(x) - b
    residual_norm = np.linalg.norm(residual)

    # Step 4.2: Compute noise floor
    noise_floor = noise_std_estimate * np.sqrt(b.size) / np.linalg.norm(b)

    # Step 4.3: Compute gate and target relative residual
    gate_rel = max(0.12, 1.10 * noise_floor + 0.02)
    target_rel = max(0.12, 0.90 * noise_floor)

    # Step 4.4: Compute complementarity
    grad = AT(residual)
    comp = np.linalg.norm(np.minimum(grad, x)) / np.max([np.linalg.norm(x), 1e-12])

    # Step 4.5: Compute score
    score = abs(residual_norm / np.linalg.norm(b) - target_rel)
    if residual_norm / np.linalg.norm(b) > gate_rel:
        score += 10 * (residual_norm / np.linalg.norm(b) - gate_rel)
    score += 2 * max(0, comp - 0.02) + 0.75 * np.mean(mask)

    return score

# Step 5: Solve the problem
x0 = np.zeros_like(b)
lambda_grid = np.logspace(-4, -1, 8)
x = solve_non_negative_l1_FISTA(A, AT, b, x0, lambda_grid)

# Step 6: Find best support
threshold_grid = np.logspace(-4, 0, 8)
mask = find_best_support(A, AT, b, x, lambda_grid, threshold_grid)

# Step 7: Debias the final solution
x = x.copy()
for step in range(420):
    grad = AT(A(x) - b)
    x[mask] = np.maximum(x[mask] - 0.98 * grad[mask], 0.0)

# Step 8: Write outputs
np.save('reconstruction.npy', x)
```