```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta

# Constants
T_FREEZE = 273.15  # K
C_ICE = 2114  # J/kg/K
L_F = 334000  # J/kg
SIGMA = 5.67e-8  # W/m^2/K^4
RHO_AIR = 1.225  # kg/m^3
C_P = 1005  # J/kg/K
R_A = 50  # s/m
K_EFF = lambda rho: 2.22e-2 + 7.75e-7 * rho**2
RHO_ICE = 917  # kg/m^3
RHO_DEST_THRESH = 150  # kg/m^3
ETA0 = 7.62e6  # Pa*s
C5 = 0.08  # 1/K
C6 = 0.023  # m^3/kg
C1_META = 2.777e-6  # 1/s
C2_META = 0.04  # 1/K

def analyze_snowpack():
    # Load data
    with open('data/snowpack_init.json', 'r') as f:
        snowpack_init = json.load(f)
    snowpack = np.array([[float(val) for val in layer.values()] for layer in snowpack_init])
    snowpack = np.swapaxes(snowpack, 0, 1)  # [layers, properties]

    forcing = pd.read_csv('data/forcing.csv', parse_dates=['time_hr'])
    forcing['time_hr'] = pd.to_datetime(forcing['time_hr'])
    forcing['t_air_C'] = forcing['t_air_K'] - T_FREEZE

    # Initialize outputs
    n_days = len(forcing) // 24
    snow_state = np.zeros((n_days, 3, 5))
    snow_summary = []

    # Time step and parameters
    dt = 3600  # 1 hour
    t = 0
    day = 0
    while t < len(forcing):
        # Extract current forcing
        current_forcing = forcing.iloc[t:t+24]
        current_forcing = current_forcing.mean()
        t_air_K = current_forcing['t_air_K']
        p_atm_Pa = current_forcing['p_atm_Pa']
        snowfall_mm_hr = current_forcing['snowfall_mm_hr']
        rainfall_mm_hr = current_forcing['rainfall_mm_hr']
        sw_in_wm2 = current_forcing['sw_in_wm2']
        lw_in_wm2 = current_forcing['lw_in_wm2']

        # Step 1: Compaction
        for layer_idx in range(3):
            z = snowpack[layer_idx, 0]
            rho = snowpack[layer_idx, 1]
            ice_mm = snowpack[layer_idx, 2]
            liq_mm = snowpack[layer_idx, 3]
            temp = snowpack[layer_idx, 4]

            # Calculate R_meta
            if temp < T_FREEZE:
                T_layer = temp
            else:
                T_layer = T_FREEZE
            T_freeze = T_FREEZE
            R_meta = -C1_META * np.exp(-C2_META * (T_freeze - T_layer)) * \
                     (np.exp(-0.046 * (rho - 150)) if rho > 150 else 1) * \
                     (1 + (liq_mm > 0.01) * 1)

            # Calculate R_overburden
            W_self = rho * z
            W_above = np.sum([snowpack[i, 1] * snowpack[i, 0] for i in range(layer_idx, 3)])
            burden_pressure = 0.5 * W_self + W_above
            eta = ETA0 * np.exp(C5 * (T_freeze - T_layer) + C6 * rho)
            R_overburden = -burden_pressure / eta

            R_total = R_meta + R_overburden
            z_new = z * (1 + dt * R_total)
            snowpack[layer_idx, 0] = z_new

            # Clip density
            ice_mass = ice_mm * 1000  # kg/m^2
            liq_mass = liq_mm * 1000  # kg/m^2
            total_mass = ice_mass + liq_mass
            rho_new = total_mass / z_new
            rho_new = np.clip(rho_new, 50, 500)
            snowpack[layer_idx, 1] = rho_new

        # Step 2: Liquid water percolation
        for layer_idx in range(2):
            z = snowpack[layer_idx, 0]
            rho = snowpack[layer_idx, 1]
            porosity = 1 - rho / RHO_ICE
            liq_capacity_mm = 0.05 * porosity * z * 1000
            liq_mass = snowpack[layer_idx, 3] * 1000  # kg/m^2
            excess_liq = max(liq_mass - liq_capacity_mm, 0)
            snowpack[layer_idx, 3] = liq_capacity_mm / 1000

            # Drain to lower layer
            if layer_idx < 2:
                snowpack[layer_idx + 1, 3] += excess_liq / 1000
            else:
                # Runoff
                snow_summary.append({
                    'day': day,
                    'swe_mm': (snowpack[0, 2] + snowpack[1, 2] + snowpack[2, 2]) * 1000,
                    'snow_depth_m': snowpack[0, 0] + snowpack[1, 0] + snowpack[2, 0],
                    'melt_mm': 0,
                    'refreeze_mm': 0,
                    'runoff_mm': excess_liq / 1000,
                    'mean_density_kg_m3': (snowpack[0, 1] + snowpack[1, 1] + snowpack[2, 1]) / 3
                })

        # Step 3: Phase change
        for layer_idx in range(3):
            z = snowpack[layer_idx, 0]
            rho = snowpack[layer_idx, 1]
            ice_mm = snowpack[layer_idx, 2]
            liq_mm = snowpack[layer_idx, 3]
            temp = snowpack[layer_idx, 4]

            ice_mass = ice_mm * 1000  # kg/m^2
            liq_mass = liq_mm * 1000  # kg/m^2

            if temp < T_FREEZE:
                cold_content = C_ICE * ice_mass * (T_FREEZE - temp)
                if liq_mass > 0:
                    refreeze_mass = min(cold_content / L_F, liq_mass)
                    ice_mass += refreeze_mass
                    liq_mass -= refreeze_mass
                    snowpack[layer_idx, 2] = ice_mass / 1000
                    snowpack[layer_idx, 3] = liq_mass / 1000
            elif temp > T_FREEZE and ice_mass > 0:
                heat_excess = C_ICE * ice_mass * (temp - T_FREEZE)
                melt_mass = min(heat_excess / L_F, ice_mass)
                ice_mass -= melt_mass
                liq_mass += melt_mass
                snowpack[layer_idx, 2] = ice_mass / 1000
                snowpack[layer_idx, 3] = liq_mass / 1000
                snowpack[layer_idx, 4] = T_FREEZE

        # Step 4: Surface energy balance
        alpha = 0.80 if snowpack[0, 3] < 0.5 else 0.55
        F_net = (1 - alpha) * sw_in_wm2 + lw_in_wm2 - SIGMA * (snowpack[0, 4] ** 4) + \
                (RHO_AIR * C_P / R_A) * (t_air_K - snowpack[0, 4])

        # Linearize radiation term
        T_old = snowpack[0, 4]
        F_net = F_net - SIGMA * (T_old ** 4)
        T_new = T_old + F_net * dt / (C_ICE * (snowpack[0, 1] * snowpack[0, 0] / 1000))
        T_new = np.clip(T_new, T_FREEZE - 1, T_FREEZE + 1)
        snowpack[0, 4] = T_new

        # Step 5: Boundary mass exchanges
        snowfall_mm = snowfall_mm_hr * dt / 3600
        rainfall_mm = rainfall_mm_hr * dt / 3600

        # Add snowfall to top layer
        rho_new = 67.92 + 51.25 * np.exp((t_air_K - T_FREEZE) / 2.59)
        rho_new = np.clip(rho_new, 50, 200)
        snowfall_mass = snowfall_mm * 1000 * rho_new / 1000  # kg/m^2
        snowpack[0, 2] += snowfall_mass / 1000
        snowpack[0, 1] = np.clip((snowpack[0, 2] + snowpack[0, 3]) * 1000 / snowpack[0, 0], 50, 500)

        # Add rainfall to top layer as liquid
        rainfall_mass = rainfall_mm * 1000  # kg/m^2
        snowpack[0, 3] += rainfall_mass / 1000

        # Save state
        snow_state[day] = snowpack
        day += 1
        t += 24

    # Save outputs
    np.save('results/snow_state.npy', snow_state)
    snow_summary_df = pd.DataFrame(snow_summary)
    snow_summary_df.to_csv('results/snow_summary.csv', index=False)

    # Plot evolution
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for i in range(3):
        axes[i].plot(snow_state[:, :, i])
        axes[i].set_title(['SWE (mm)', 'Snow Depth (m)', 'Mean Density (kg/m^3)'][i])
    plt.tight_layout()
    plt.savefig('results/snow_evolution.png')
    plt.close()

analyze_snowpack()
```