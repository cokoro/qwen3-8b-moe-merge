import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import json

# Constants
c_ice = 2.114e3  # J/kg/K
L_f = 334000  # J/kg
sigma = 5.67e-8  # W/m2/K4
rho_air = 1.225  # kg/m3
c_p = 1005  # J/kg/K
r_a = 50  # s/m
T_freeze = 273.15  # K
rho_ice = 917  # kg/m3
eta0 = 7.62e6  # Pa s
c5 = 0.08  # 1/K
c6 = 0.023  # m3/kg
c1_meta = 2.777e-6  # 1/s
c2_meta = 0.04  # 1/K
rho_dest_thresh = 150  # kg/m3
theta_irr = 0.05
rho_new_min = 50
rho_new_max = 200

# Load data
snowpack_init = json.load(open('data/snowpack_init.json'))
forcing = pd.read_csv('data/forcing.csv', parse_dates=['time_hr'])

# Initial state
layers = np.array([[
    snowpack_init[0]['thickness_m'],
    snowpack_init[0]['density_kg_m3'],
    snowpack_init[0]['ice_mm'],
    snowpack_init[0]['liq_mm'],
    snowpack_init[0]['temp_K'],
], [
    snowpack_init[1]['thickness_m'],
    snowpack_init[1]['density_kg_m3'],
    snowpack_init[1]['ice_mm'],
    snowpack_init[1]['liq_mm'],
    snowpack_init[1]['temp_K'],
], [
    snowpack_init[2]['thickness_m'],
    snowpack_init[2]['density_kg_m3'],
    snowpack_init[2]['ice_mm'],
    snowpack_init[2]['liq_mm'],
    snowpack_init[2]['temp_K'],
]], dtype=np.float64)

# Simulation
n_days = len(forcing) // 24
output = []
snow_state = []
snow_summary = []

for day_idx in range(n_days):
    # Time step
    dt = 3600  # s
    time = forcing.iloc[day_idx * 24].time_hr
    t_air_K = forcing.iloc[day_idx * 24].t_air_K
    p_atm_Pa = forcing.iloc[day_idx * 24].p_atm_Pa
    snowfall_mm_hr = forcing.iloc[day_idx * 24].snowfall_mm_hr
    rainfall_mm_hr = forcing.iloc[day_idx * 24].rainfall_mm_hr
    sw_in_wm2 = forcing.iloc[day_idx * 24].sw_in_wm2
    lw_in_wm2 = forcing.iloc[day_idx * 24].lw_in_wm2

    # Surface layer
    layer0 = layers[0]
    layer1 = layers[1]
    layer2 = layers[2]

    # Step 1: Compaction
    for i in range(3):
        layer = layers[i]
        z = layer[0]
        rho = layer[1]
        ice = layer[2]
        liq = layer[3]
        T = layer[4]

        # Meta rate
        if rho > rho_dest_thresh:
            density_factor = np.exp(-0.046 * (rho - 150))
        else:
            density_factor = 1.0
        if liq > 0.01:
            liquid_factor = 2.0
        else:
            liquid_factor = 1.0
        R_meta = c1_meta * np.exp(-c2_meta * (T_freeze - T)) * density_factor * liquid_factor

        # Overburden pressure
        W_self = rho * z * 1000  # kg/m2
        W_above = (layers[0][1] * layers[0][0] + layers[1][1] * layers[1][0]) * 1000
        burden_pressure = 0.5 * W_self + W_above  # kg/m2
        R_overburden = -burden_pressure / eta(T, rho)

        R_total = R_meta + R_overburden
        dz = z * dt * R_total
        z_new = z + dz
        layers[i][0] = z_new

        # Density update
        ice_kg_m2 = ice * 1000  # kg/m2
        liq_kg_m2 = liq * 1000  # kg/m2
        rho_new = (ice_kg_m2 + liq_kg_m2) / z_new * 1000  # kg/m3
        rho_new = np.clip(rho_new, 50, 500)
        layers[i][1] = rho_new

    # Step 2: Liquid percolation
    for i in range(2):
        layer = layers[i]
        z = layer[0]
        rho = layer[1]
        ice = layer[2]
        liq = layer[3]
        T = layer[4]

        porosity = 1 - rho / rho_ice
        liq_capacity = theta_irr * porosity * z * 1000  # mm
        liq_drain = max(liq - liq_capacity, 0)
        layers[i][3] -= liq_drain
        layers[i+1][3] += liq_drain
    runoff_mm = layers[2][3]
    layers[2][3] = 0

    # Step 3: Phase change
    for i in range(3):
        layer = layers[i]
        z = layer[0]
        rho = layer[1]
        ice = layer[2]
        liq = layer[3]
        T = layer[4]

        # Cold content
        if T < T_freeze:
            cold_content = c_ice * ice * (T_freeze - T)
        else:
            cold_content = 0
        heat_excess = c_ice * ice * (T - T_freeze)
        if T > T_freeze and ice > 0:
            melt_mass = min(heat_excess / L_f, ice)
            ice -= melt_mass
            liq += melt_mass
            T = T_freeze
        elif T < T_freeze and liq > 0:
            refreeze_mass = min(cold_content / L_f, liq)
            ice += refreeze_mass
            liq -= refreeze_mass
        layer[2] = ice
        layer[3] = liq
        layer[4] = T

    # Step 4: Surface energy balance
    layer0 = layers[0]
    z0 = layer0[0]
    T0 = layer0[4]
    alpha = 0.8 if liq < 0.5 else 0.55
    F_net = (1 - alpha) * sw_in_wm2 + lw_in_wm2 - sigma * T0**4 + rho_air * c_p / r_a * (t_air_K - T0)
    # Temperature update
    # Use phase change to handle latent heat
    if T0 < T_freeze and liq > 0:
        refreeze_mass = min(c_ice * liq * (T_freeze - T0) / L_f, liq)
        liq -= refreeze_mass
        T0 = T_freeze
    elif T0 > T_freeze and ice > 0:
        melt_mass = min(c_ice * ice * (T0 - T_freeze) / L_f, ice)
        ice -= melt_mass
        liq += melt_mass
        T0 = T_freeze
    layer0[2] = ice
    layer0[3] = liq
    layer0[4] = T0

    # Step 5: Boundary mass exchanges
    # Snowfall
    if snowfall_mm_hr > 0:
        rho_new = np.clip(67.92 + 51.25 * np.exp((t_air_K - 273.15)/2.59), rho_new_min, rho_new_max)
        snowfall_kg_m2 = snowfall_mm_hr * 1000 * rho_new / 1000
        layers[0][2] += snowfall_kg_m2 / 1000
        layers[0][3] += snowfall_kg_m2 / 1000
        layers[0][1] = rho_new
    # Rainfall
    if rainfall_mm_hr > 0:
        layers[0][3] += rainfall_mm_hr * 1000 / 1000

    # Save state
    snow_state.append(np.stack([layer[0], layer[1], layer[2], layer[3], layer[4]] for layer in layers))
    snow_summary.append([day_idx, 1000 * (layers[0][2] + layers[0][3] + layers[1][2] + layers[1][3] + layers[2][2] + layers[2][3]), layers[0][0] + layers[1][0] + layers[2][0], runoff_mm, 0, 0, layers[0][1] + layers[1][1] + layers[2][1]])

# Save output
np.save('results/snow_state.npy', np.array(snow_state))
pd.DataFrame(snow_summary, columns=['day', 'swe_mm', 'snow_depth_m', 'melt_mm', 'refreeze_mm', 'runoff_mm', 'mean_density_kg_m3']).to_csv('results/snow_summary.csv', index=False)

# Plot evolution
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
for i in range(3):
    axs[i].plot(np.arange(n_days), snow_state[:, i, 0])
    axs[i].set_title(['SWE', 'Snow depth', 'Mean density'][i])
    axs[i].grid(True)
plt.tight_layout()
plt.savefig('results/snow_evolution.png')
plt.close()