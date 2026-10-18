import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data_path = os.path.join('data', 'snowpack_init.json')
forcing_path = os.path.join('data', 'forcing.csv')

# Parse initial state
initial_state = pd.read_json(data_path)
layer_data = initial_state.to_dict('records')

# Parse forcing data
forcing = pd.read_csv(forcing_path, parse_dates=['time_hr'])
forcing['time_hr'] = pd.to_datetime(forcing['time_hr']).dt.total_seconds().div(3600)
forcing['t_air_C'] = forcing['t_air_K'].subtract(273.15)

# Constants
T_FREEZE = 273.15  # K
RHO_ICE = 917  # kg/m^3
C_ICE = 2114  # J/kg/K
L_F = 334000  # J/kg
SIGMA = 5.67e-8  # W/m^2/K^4
RHO_AIR = 1.225  # kg/m^3
C_P = 1005  # J/kg/K
R_A = 50  # s/m
K_EFF = lambda rho: 2.22e-2 + 7.75e-7 * rho**2

# Parameters
c1_meta = 2.777e-6  # 1/s
c2_meta = 0.04  # 1/K
eta0 = 7.62e6  # Pa s
c5 = 0.08  # 1/K
c6 = 0.023  # m^3/kg

# Initialize state
n_days = 100
state = np.zeros((n_days, 3, 5))
state[0, :, :] = [d['thickness_m'], d['density_kg_m3'], d['ice_mm'], d['liq_mm'], d['temp_K'] for d in layer_data]
snow_state = state.copy()

# Process each time step
for i in range(1, n_days):
    # Step 1: Compaction
    for j in range(3):
        z = state[i-1, j, 0]
        rho = state[i-1, j, 1]
        ice = state[i-1, j, 2]
        liq = state[i-1, j, 3]
        T_layer = state[i-1, j, 4]
        
        # Compute rates
        R_meta = c1_meta * np.exp(-c2_meta * (T_FREEZE - T_layer)) * \
                 (np.exp(-0.046 * (rho - 150)) if rho > 150 else 1) * \
                 (1 + (liq > 0.01) * 1)
        R_overburden = -forcing['burden_pressure_kg_m2'][i] / (eta0 * np.exp(c5 * (T_FREEZE - T_layer) + c6 * rho))
        R_total = R_meta + R_overburden
        
        # Update thickness
        z_new = z * (1 + forcing['dt'][i] * R_total)
        z_new = np.clip(z_new, 1e-3, 10)  # Avoid numerical issues
        
        # Update density (ice mass is conserved)
        rho_new = ice * 1000 / z_new  # kg/m^2 to kg/m^3
        rho_new = np.clip(rho_new, 50, 500)  # kg/m^3 bounds
        
        # Update state
        state[i, j, 0] = z_new
        state[i, j, 1] = rho_new
        state[i, j, 2] = ice * 1000 / z_new  # kg/m^2 to mm SWE-equivalent
        state[i, j, 3] = liq * 1000 / z_new  # kg/m^2 to mm
    
    # Step 2: Liquid percolation
    for j in range(3):
        z = state[i, j, 0]
        rho = state[i, j, 1]
        ice = state[i, j, 2]
        liq = state[i, j, 3]
        T_layer = state[i, j, 4]
        
        porosity = 1 - rho / RHO_ICE
        liq_capacity = theta_irr * porosity * z * 1000  # mm of water
        
        # Drain to next layer
        if j < 2:
            drain = np.max([liq - liq_capacity, 0])
            state[i, j, 3] = liq - drain
            state[i, j+1, 3] += drain
        else:
            runoff = np.max([liq - liq_capacity, 0])
            state[i, j, 3] = liq - runoff
            runoff_mm = runoff
    
    # Step 3: Phase change
    for j in range(3):
        z = state[i, j, 0]
        rho = state[i, j, 1]
        ice = state[i, j, 2]
        liq = state[i, j, 3]
        T_layer = state[i, j, 4]
        
        # Compute cold content and heat excess
        if T_layer < T_FREEZE:
            cold_content = C_ICE * ice * (T_FREEZE - T_layer)
        else:
            heat_excess = C_ICE * ice * (T_layer - T_FREEZE)
        
        # Melt/refreeze
        if T_layer > T_FREEZE and ice > 0:
            melt_mass = np.min([heat_excess / L_F, ice])
            ice -= melt_mass
            liq += melt_mass
            T_layer = T_FREEZE
        elif T_layer < T_FREEZE and liq > 0:
            refreeze_mass = np.min([cold_content / L_F, liq])
            ice += refreeze_mass
            liq -= refreeze_mass
        
        # Update state
        state[i, j, 2] = ice
        state[i, j, 3] = liq
        state[i, j, 4] = T_layer
    
    # Step 4: Surface energy balance
    t_air = forcing['t_air_K'][i]
    alpha = 0.8 if state[i, 0, 3] < 0.5 else 0.55
    F_net = (1 - alpha) * forcing['sw_in_wm2'][i] + forcing['lw_in_wm2'][i] - \
            SIGMA * state[i, 0, 4]**4 + RHO_AIR * C_P / R_A * (t_air - state[i, 0, 4])
    
    # Update surface temperature
    if state[i, 0, 3] < T_FREEZE:
        state[i, 0, 4] = np.clip(state[i, 0, 4] + F_net * forcing['dt'][i] / C_ICE * 1000, 250, 300)
    else:
        state[i, 0, 4] = np.clip(state[i, 0, 4] + F_net * forcing['dt'][i] / C_ICE * 1000, 270, 300)
    
    # Step 5: Mass exchanges
    # Snowfall
    rho_new = np.clip(67.92 + 51.25 * np.exp(forcing['t_air_C'][i] / 2.59), 50, 200)
    snowfall = forcing['snowfall_mm_hr'][i] * 1000 * rho_new / 1000  # kg/m^2
    state[i, 0, 2] += snowfall
    state[i, 0, 1] = np.clip(snowfall / state[i, 0, 0], 50, 500)
    
    # Rainfall
    rainfall = forcing['rainfall_mm_hr'][i] * 1000  # kg/m^2
    state[i, 0, 3] += rainfall
    
    # Update snow_state
    snow_state[i, :, :] = state[i, :, :]

# Save outputs
np.save('results/snow_state.npy', snow_state)

# Compute summary statistics
summary = []
for i in range(n_days):
    swe = np.sum(state[i, :, 2])  # mm SWE
    depth = state[i, 0, 0]  # m
    melt = np.sum(state[i, :, 2] - state[i-1, :, 2]) if i > 0 else 0
    refreeze = np.sum(state[i-1, :, 2] - state[i, :, 2]) if i > 0 else 0
    runoff = state[i, 2, 3]  # mm
    mean_density = np.mean(state[i, :, 1])
    summary.append([i, swe, depth, melt, refreeze, runoff, mean_density])

summary_df = pd.DataFrame(summary, columns=['day', 'swe_mm', 'snow_depth_m', 'melt_mm', 'refreeze_mm', 'runoff_mm', 'mean_density_kg_m3'])
summary_df.to_csv('results/snow_summary.csv', index=False)

# Plot evolution
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
for ax, label in zip(axes, ['SWE (mm)', 'Snow depth (m)', 'Mean density (kg/m^3)']):
    ax.plot(summary_df['day'], summary_df[label])
    ax.set_ylabel(label)
    ax.set_xlabel('Day')
plt.tight_layout()
plt.savefig('results/snow_evolution.png')
plt.close()