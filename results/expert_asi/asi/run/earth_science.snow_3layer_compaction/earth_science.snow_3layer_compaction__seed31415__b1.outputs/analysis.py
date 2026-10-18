import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load data
snow_init = np.load('data/snowpack_init.npy')
forcing = pd.read_csv('data/forcing.csv', parse_dates=['time_hr'])

# Constants
T_freeze = 273.15  # K
c_ice = 2114  # J/kg/K
L_f = 334000  # J/kg
sigma = 5.67e-8  # W/m^2/K^4
rho_air = 1.225  # kg/m^3
c_p = 1005  # J/kg/K
r_a = 50  # s/m
k_ref = 2.22e-2  # W/m/K
rho_ice = 917  # kg/m^3

def step(dt_sec, state, forcing_row):
    dt = dt_sec / 3600  # hours
    t_air = forcing_row['t_air_K']
    snowfall = forcing_row['snowfall_mm_hr'] * dt
    rainfall = forcing_row['rainfall_mm_hr'] * dt

    # Step 1: Compaction
    new_thickness = np.copy(state['thickness'])
    new_density = np.copy(state['density'])
    new_ice = np.copy(state['ice'])
    new_liq = np.copy(state['liq'])
    new_temp = np.copy(state['temp'])

    for i in range(3):
        z = state['thickness'][i]
        rho = state['density'][i]
        ice_mm = state['ice'][i]
        liq_mm = state['liq'][i]
        T = state['temp'][i]

        # Metamorphism rate
        if rho < 150:
            density_factor = 1.0
        else:
            density_factor = np.exp(-0.046 * (rho - 150))
        if liq_mm > 0.01:
            liquid_factor = 2.0
        else:
            liquid_factor = 1.0

        # Overburden pressure
        W_self = rho * z * 1000  # kg/m^2
        W_above = np.sum(state['density'][1:] * state['thickness'][1:]) * 1000
        burden = (W_self + W_above) / 2  # kg/m^2
        eta = 7.62e6 * np.exp(0.08 * (T_freeze - T) + 0.023 * rho)
        R_overburden = -burden / eta

        # Metamorphism rate
        c1_meta = 2.777e-6
        c2_meta = 0.04
        R_meta = -c1_meta * np.exp(-c2_meta * (T_freeze - T)) * density_factor * liquid_factor
        R_total = R_meta + R_overburden

        # Update thickness
        z_new = z * (1 + R_total * dt)
        new_thickness[i] = z_new

        # Clip density
        ice_mass = ice_mm * 1000  # kg/m^2
        rho_new = (ice_mass / z_new) * 1000  # kg/m^3
        rho_new = np.clip(rho_new, 50, 500)
        new_density[i] = rho_new

    # Step 2: Liquid drainage
    porosity = 1 - state['density'] / rho_ice
    liq_capacity = theta_irr * porosity * state['thickness'] * 1000  # mm
    excess_liq = new_liq - liq_capacity
    if excess_liq > 0:
        new_liq[0] = liq_capacity[0]
        runoff = excess_liq[0] / 1000  # m
        for i in range(1, 3):
            new_liq[i] = np.clip(new_liq[i] + excess_liq[i] / 1000, 0, liq_capacity[i])
        runoff = excess_liq[0] / 1000  # m

    # Step 3: Phase change
    for i in range(3):
        T = new_temp[i]
        ice_mm = new_ice[i]
        liq_mm = new_liq[i]
        if T > T_freeze and ice_mm > 0:
            cold_content = c_ice * ice_mm * (T_freeze - T)
            heat_excess = c_ice * ice_mm * (T - T_freeze)
            melt_mass = min(heat_excess / L_f, ice_mm)
            new_ice[i] -= melt_mass
            new_liq[i] += melt_mass
            new_temp[i] = T_freeze
        elif T < T_freeze and liq_mm > 0:
            cold_content = c_ice * ice_mm * (T_freeze - T)
            refreeze_mass = min(cold_content / L_f, liq_mm)
            new_ice[i] += refreeze_mass
            new_liq[i] -= refreeze_mass

    # Step 4: Surface energy balance
    alpha = 0.8 if new_liq[0] < 0.5 else 0.55
    F_net = (1 - alpha) * forcing_row['sw_in_wm2'] + forcing_row['lw_in_wm2'] \
            - sigma * new_temp[0] ** 4 + rho_air * c_p / r_a * (t_air - new_temp[0])
    C = rho_ice * (1 - new_density[0]/rho_ice) * 1000  # kg/m^2
    new_temp[0] = new_temp[0] + F_net * dt / C

    # Step 5: Mass addition
    rho_new_snow = np.clip(67.92 + 51.25 * np.exp((t_air - 273.15)/2.59), 50, 200)
    new_ice[0] += snowfall * rho_new_snow / 1000  # m
    new_liq[0] += rainfall * 1000  # m

    return {
        'thickness': new_thickness,
        'density': new_density,
        'ice': new_ice,
        'liq': new_liq,
        'temp': new_temp
    }

# Simulation loop
state = {'thickness': snow_init[:, 0], 'density': snow_init[:, 1], 'ice': snow_init[:, 2], 'liq': snow_init[:, 3], 'temp': snow_init[:, 4]}
output = []
total_swe = 0
total_runoff = 0

for i, row in forcing.iterrows():
    dt_sec = 3600  # 1 hour
    state = step(dt_sec, state, row)
    output.append({
        'day': i // 24,
        'swe_mm': sum(state['ice'] * 1000),
        'depth_m': state['thickness'][0],
        'melt_mm': np.nan,
        'refreeze_mm': np.nan,
        'runoff_mm': state['liq'][0] * 1000,
        'mean_density': np.mean(state['density'])
    })
    total_swe += sum(state['ice'] * 1000)
    total_runoff += state['liq'][0] * 1000

np.save('results/snow_state.npy', np.array(output))
pd.DataFrame(output).to_csv('results/snow_summary.csv', index=False)

# Plotting
plt.figure(figsize=(12, 6))
for i in range(3):
    plt.plot([x['day'] for x in output], [x['depth_m'] for x in output], label=f'Layer {i}')
plt.legend()
plt.xlabel('Day')
plt.ylabel('Depth (m)')
plt.savefig('results/snow_evolution.png')
plt.close()