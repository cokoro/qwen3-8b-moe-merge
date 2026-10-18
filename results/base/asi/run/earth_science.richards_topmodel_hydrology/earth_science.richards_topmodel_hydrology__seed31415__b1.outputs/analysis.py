import numpy as np
import json
import csv
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Load data
with open('data/soil_profile.json') as f:
    soil_profile = json.load(f)
n_layers = soil_profile['n_layers']
layer_thicknesses_m = np.array(soil_profile['layer_thicknesses_m'])
layer_centers_m = np.array(soil_profile['layer_centers_m'])
theta_sat = np.array(soil_profile['theta_sat'])
psi_sat_m = np.array(soil_profile['psi_sat_m'])
b_param = np.array(soil_profile['b_param'])
k_sat_m_s = np.array(soil_profile['k_sat_m_s'])
theta_init = np.array(soil_profile['theta_init'])

with open('data/forcing.csv') as f:
    reader = csv.DictReader(f)
    forcing = [dict(row) for row in reader]
forcing = np.array([(int(row['day']), float(row['precip_mm_day']), float(row['pet_mm_day'])) for row in forcing])

with open('data/site.json') as f:
    site = json.load(f)
topo_index = site['topo_index']
init_water_table_m = site['init_water_table_m']
dt_seconds = site['dt_seconds']
n_days = site['n_days']

# Constants
RUNOFF_DECAY = 2.0
K_baseflow_mm_s = 5e-3
specific_yield = 0.20
theta_residual = 0.02
theta_wilt = 0.10  # approximate theta at -150 m
theta_fc = 0.30   # approximate theta at -3.3 m
root_frac = np.array([0.4, 0.3, 0.2, 0.1])  # root distribution

# Initialize variables
theta = theta_init.copy()
water_table_depth = init_water_table_m
aquifer_storage_mm = 0.0
soil_moisture = np.zeros((n_days, n_layers))
fluxes = np.zeros((n_days, 6))  # infiltration, runoff, recharge, baseflow, drainage, et
cumulative_balance = np.zeros(n_days)

# Time stepping
for day in range(n_days):
    precip_mm_day, pet_mm_day = forcing[day][1], forcing[day][2]
    day_start = day * 24 * 3600
    day_end = (day + 1) * 24 * 3600
    dt = dt_seconds
    
    # Surface runoff
    sat_frac = 0.5 * np.exp(-RUNOFF_DECAY * water_table_depth) * np.exp(topo_index / 10.0)
    sat_frac = np.clip(sat_frac, 0, 1)
    runoff_surface = sat_frac * precip_mm_day
    infiltration = (1 - sat_frac) * precip_mm_day
    
    # ET calculation
    theta_wilt = theta_at_psi(-150, theta_sat, psi_sat_m, b_param)
    theta_fc = theta_at_psi(-3.3, theta_sat, psi_sat_m, b_param)
    beta = np.zeros(n_layers)
    for l in range(n_layers):
        beta[l] = (theta[l] - theta_wilt) / (theta_fc - theta_wilt) * root_frac[l]
    beta = np.clip(beta, 0, 1)
    et_actual = beta * pet_mm_day
    
    # Richards equation setup
    # Compute hydraulic properties
    S = np.maximum(theta / theta_sat, 0.01)
    psi = psi_sat_m * S**(-b_param)
    K = k_sat_m_s * S**(2 * b_param + 3)
    D = -K * b_param * psi / theta
    
    # Interface properties (harmonic mean for K, arithmetic mean for D)
    K_interface = np.zeros(n_layers - 1)
    D_interface = np.zeros(n_layers - 1)
    for l in range(n_layers - 1):
        K_interface[l] = 2 / (1 / K[l] + 1 / K[l + 1])
        D_interface[l] = (D[l] + D[l + 1]) / 2
    
    # Tridiagonal matrix for Richards equation
    # [I/dt - L_implicit] * theta_new = theta_old / dt + b_explicit
    # L_implicit is tridiagonal with diagonals a, b, c
    a = np.zeros(n_layers)
    b = np.zeros(n_layers)
    c = np.zeros(n_layers)
    for l in range(n_layers):
        if l == 0:
            a[l] = 0
        else:
            a[l] = -K_interface[l - 1] / layer_thicknesses_m[l - 1]
        if l == n_layers - 1:
            c[l] = 0
        else:
            c[l] = -K_interface[l] / layer_thicknesses_m[l]
        b[l] = 1 / dt + K[l] / layer_thicknesses_m[l] + D[l] / layer_thicknesses_m[l]
    
    # Right-hand side
    rhs = theta / dt
    rhs[0] += infiltration / 1000  # convert mm/day to m/day
    rhs[-1] -= (K[-1] * (psi[-1] - 0) / (water_table_depth - layer_centers_m[-1]) + K[-1]) * dt / 1000  # drainage
    rhs[-1] += (K_baseflow_mm_s * 1e-3 * np.exp(-topo_index) * np.exp(-RUNOFF_DECAY * water_table_depth)) * dt / 1000  # baseflow
    
    # Solve tridiagonal system
    banded = np.zeros((3, n_layers))
    banded[0, 1:] = a[1:]
    banded[1, :] = b
    banded[2, :-1] = c[:-1]
    theta_new = solve_banded((1, 1), banded, rhs)
    
    # Clip theta to [theta_residual, theta_sat]
    theta_new = np.clip(theta_new, theta_residual, theta_sat)
    theta = theta_new
    
    # Compute fluxes
    recharge = (K[-1] * (0 - psi[-1]) / (water_table_depth - layer_centers_m[-1]) + K[-1]) * 1000 / dt
    recharge = np.clip(recharge, -2e-6, 2e-6)
    baseflow = K_baseflow_mm_s * np.exp(-topo_index) * np.exp(-RUNOFF_DECAY * water_table_depth)
    baseflow = baseflow * 1e-3  # convert mm/s to m/s
    net_flux_mm = (recharge - baseflow) * dt / 1000  # convert to mm/day
    aquifer_storage_mm += net_flux_mm
    water_table_depth = np.clip(water_table_depth - net_flux_mm / specific_yield / 1000, 0, 5)
    
    # Store results
    soil_moisture[day] = theta
    fluxes[day] = [infiltration, runoff_surface, recharge * 1000, baseflow * 1000, (recharge - baseflow) * 1000, et_actual]
    cumulative_balance[day] = (precip_mm_day - et_actual - runoff_surface - baseflow * 1000 - net_flux_mm)
    
    # Mass closure check
    total_water = np.sum(theta * layer_thicknesses_m) + aquifer_storage_mm
    total_water_initial = np.sum(theta_init * layer_thicknesses_m) + aquifer_storage_mm
    delta_storage = total_water - total_water_initial
    cumulative_balance[day] -= delta_storage

# Save outputs
np.save('results/soil_moisture.npy', soil_moisture)
with open('results/water_table.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['day', 'water_table_m', 'aquifer_storage_mm'])
    for day in range(n_days):
        writer.writerow([day, water_table_depth, aquifer_storage_mm])
with open('results/fluxes.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['day', 'infiltration', 'runoff_surface', 'recharge', 'baseflow', 'drainage_bot', 'et_actual'])
    for day in range(n_days):
        writer.writerow([day, fluxes[day][0], fluxes[day][1], fluxes[day][2], fluxes[day][3], fluxes[day][4], fluxes[day][5]])

# Plot water balance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.plot(cumulative_balance)
ax1.set_title('Cumulative Water Balance')
ax1.set_xlabel('Day')
ax1.set_ylabel('Balance (mm)')
ax2.plot(soil_moisture[::n_days//2, :], 'o-')
ax2.set_title('Soil Moisture Profile')
ax2.set_xlabel('Layer')
ax2.set_ylabel('Moisture (m^3/m^3)')
plt.tight_layout()
plt.savefig('results/water_balance.png')
plt.close()