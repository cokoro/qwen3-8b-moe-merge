```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Read initial state and forcing
snow_init = pd.read_json('data/snowpack_init.json', orient='index')
forcing = pd.read_csv('data/forcing.csv', index_col=0)

# Parameters
c1_meta = 2.777e-6
c2_meta = 0.04
eta0 = 7.62e6
c5 = 0.08
c6 = 0.023
rho_ice = 917
c_ice = 2.114e3  # J/kg/K
L_f = 334000  # J/kg
sigma = 5.67e-8
rho_air = 1.225
c_p = 1005
r_a = 50
k_eff = lambda rho: 2.22e-2 + 7.75e-7 * rho**2
T_freeze = 273.15

# Constants
dt = 3600  # seconds per step
dt_surf = 100  # substep for surface
n_days = len(forcing)

# Initialize state
state = np.zeros((n_days, 3, 5))
state[0] = snow_init.values

# Precompute air temps and snowfall
air_temp = forcing['t_air_K'].values
snowfall_mm = forcing['snowfall_mm_hr'].values
rainfall_mm = forcing['rainfall_mm_hr'].values

# Step through each day
for day in range(n_days):
    # Step 1: Compaction
    for l in range(3):
        z = state[day, l, 0]
        rho = state[day, l, 1]
        ice = state[day, l, 2]
        liq = state[day, l, 3]
        T = state[day, l, 4]
        
        # Compute R_meta
        if T < T_freeze:
            rho_factor = 1.0 if rho < 150 else np.exp(-0.046 * (rho - 150))
        else:
            rho_factor = 1.0
        if liq > 0.01:
            liq_factor = 2.0
        else:
            liq_factor = 1.0
        R_meta = -c1_meta * np.exp(-c2_meta * (T_freeze - T)) * rho_factor * liq_factor
        
        # Compute R_overburden
        W_self = rho * z * 1000  # kg/m^2
        W_above = sum(state[day, i, 1] * state[day, i, 0] * 1000 for i in range(l+1, 3))
        burden = 0.5 * W_self + W_above
        eta = eta0 * np.exp(c5 * (T_freeze - T) + c6 * rho)
        R_overburden = -burden / eta
        
        R_total = R_meta + R_overburden
        dz = z * dt * R_total
        z_new = z + dz
        
        # Update ice mass to maintain constant mass
        ice_new = ice * (z / z_new)
        # Clip density to [50, 500] kg/m^3
        rho_new = max(50, min(500, rho * z / z_new))
        
        state[day+1, l, 0] = z_new
```