```python
import numpy as np
from scipy.fft import dst, idst

# Constants
N = 135
dx = 1.0 / N
inv_dx = N
CFL = 1.8797
dt_frame = 0.1
num_frames = 108

# Load initial conditions
u_x_np = np.load("data/initial_u_x.npy")
u_y_np = np.load("data/initial_u_y.npy")

# Step 1: Compute initial nodal vorticity
w_n = np.zeros((N+1, N+1), dtype=np.float32)
for i in range(N):
    for j in range(N):
        w_n[i, j] = (u_y_np[i, j] - u_y_np[i-1, j] - u_x_np[i, j
```