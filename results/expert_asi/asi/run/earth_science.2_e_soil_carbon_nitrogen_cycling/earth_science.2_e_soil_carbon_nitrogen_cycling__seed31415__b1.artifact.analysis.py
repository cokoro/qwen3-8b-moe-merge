import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
import json

# Read inputs
soil_props = json.load(open('data/soil_properties.json'))
climate = pd.read_csv('data/climate_forcing.csv')
litter = pd.read_csv('data/litter_input.csv')

# Initialize model state
n_days = len(climate)
pool_names = soil_props['pool_names']
C_pools = np.zeros((n_days, len(pool_names)))
N_pools = np.zeros((n_days, len(pool_names)))
C_pools[0, 0] = soil_props['initial_c']
N_pools[0, 0] = soil_props['initial_n']

# Process climate data
def get_scalar(climate, day_idx, field='soil_moisture_m3m3'):
    moisture = climate[day_idx][field]
    if moisture < soil_props['wilting_point']:
        return 0.0
    elif moisture > soil_props['field_capacity']:
        return 1.0
    else:
        return (moisture - soil_props['wilting_point']) / (soil_props['field_capacity'] - soil_props['wilting_point'])

# Define ODE system
def ode_system(t, y, params):
    C = y[:len(pool_names)]
    N = y[len(pool_names):]
    temp = params['temp'][int(t)]
    moisture = params['moisture'][int(t)]
    q10 = params['q10']
    t_ref = 20.0
    temp_scalar = q10 ** ((temp - t_ref) / 10)
    moisture_scalar = get_scalar(climate, int(t), 'soil_moisture_m3m3')
    total_scalar = temp_scalar * moisture_scalar

    # Decomposition rates (yr⁻¹)
    k = np.array([
        14.8,  # METL
        3.9,   # STRL
        7.3 * (1 - 0.5 * soil_props['clay_fraction']),  # Active
        0.20,  # Slow
        0.0045 # Passive
    ]) * total_scalar

    # Transfer pathways
    # METL -> Active + CO2
    metl_to_active = k[0] * C[0]
    active_c = C[2] + metl_to_active * 0.55
    active_n = N[2] + (metl_to_active * 0.55 * soil_props['metl_c_n_ratio']) / soil_props['metl_c_n_ratio']
    C[0] -= metl_to_active
    C[2] += active_c
    N[0] -= metl_to_active * 0.55 * soil_props['metl_c_n_ratio']
    N[2] += active_n

    # STRL -> Active + Slow + CO2
    strl_to_active = k[1] * C[1] * 0.7
    strl_to_slow = k[1] * C[1] * 0.3
    active_c += strl_to_active * 0.55
    slow_c = C[3] + strl_to_slow * 0.55
    active_n += strl_to_active * 0.55 * soil_props['strl_c_n_ratio'] / soil_props['strl_c_n_ratio']
    slow_n = N[3] + strl_to_slow * 0.55 * soil_props['strl_c_n_ratio'] / soil_props['strl_c_n_ratio']
    C[1] -= strl_to_active + strl_to_slow
    C[2] += active_c
    C[3] += slow_c
    N[1] -= strl_to_active * 0.55 * soil_props['strl_c_n_ratio']
    N[3] += slow_n

    # Active -> Slow + Passive + CO2
    active_to_slow = k[2] * C[2] * (1 - 0.68 * soil_props['clay_fraction'])
    active_to_passive = k[2] * C[2] * 0.68 * soil_props['clay_fraction']
    slow_c += active_to_slow * 0.55
    passive_c = C[4] + active_to_passive * 0.55
    slow_n += active_to_slow * 0.55 * soil_props['active_c_n_ratio'] / soil_props['active_c_n_ratio']
    passive_n = N[4] + active_to_passive * 0.55 * soil_props['active_c_n_ratio'] / soil_props['active_c_n_ratio']
    C[2] -= active_to_slow + active_to_passive
    C[3] += slow_c
    C[4] += passive_c
    N[2] -= active_to_slow * 0.55 * soil_props['active_c_n_ratio']
    N[4] += passive_n

    # Slow -> Active + Passive + CO2
    slow_to_active = k[3] * C[3] * 0.6
    slow_to_passive = k[3] * C[3] * 0.4
    active_c += slow_to_active * 0.55
    passive_c += slow_to_passive * 0.55
    active_n += slow_to_active * 0.55 * soil_props['slow_c_n_ratio'] / soil_props['slow_c_n_ratio']
    passive_n += slow_to_passive * 0.55 * soil_props['slow_c_n_ratio'] / soil_props['slow_c_n_ratio']
    C[3] -= slow_to_active + slow_to_passive
    C[2] += active_c
    C[4] += passive_c
    N[3] -= slow_to_active * 0.55 * soil_props['slow_c_n_ratio']
    N[4] += passive_n

    # Passive -> Active + CO2
    passive_to_active = k[4] * C[4] * 0.2
    active_c += passive_to_active * 0.55
    passive_c = C[4] - passive_to_active
    active_n += passive_to_active * 0.55 * soil_props['passive_c_n_ratio'] / soil_props['passive_c_n_ratio']
    C[4] -= passive_to_active
    C[2] += active_c
    N[4] -= passive_to_active * 0.55 * soil_props['passive_c_n_ratio']
    N[2] += active_n

    # Net N mineralization
    net_n_mineralization = 0.0
    for i in range(len(pool_names)):
        if N[i] < 0:
            net_n_mineralization += -N[i]
            N[i] = 0.0

    return np.concatenate([C, N])

# Run simulation
params = {
    'temp': [climate.loc[i, 'temperature_C'] for i in range(n_days)],
    'moisture': [climate.loc[i, 'soil_moisture_m3m3'] for i in range(n_days)],
    'q10': soil_props['q10']
}
sol = solve_ivp(ode_system, [0, n_days-1], np.concatenate([C_pools[0], N_pools[0]]), 
               t_eval=np.arange(n_days), method='BDF', rtol=1e-6, atol=1e-8, args=(params,))

# Save outputs
os.makedirs('results', exist_ok=True)
np.save('results/carbon_pools.npy', sol.y[:len(pool_names)].T)
np.save('results/nitrogen_pools.npy', sol.y[len(pool_names):].T)

fluxes = pd.DataFrame({
    'day': np.arange(n_days),
    'soil_respiration_gC': [sum(sol.y[:len(pool_names), i] * 0.55 for i in range(len(pool_names))) for _ in range(n_days)],
    'net_n_mineralization_gN': [sum(sol.y[len(pool_names):, i] for i in range(len(pool_names))) for _ in range(n_days)],
    'n_immobilization_gN': [sum(sol.y[len(pool_names):, i] for i in range(len(pool_names))) for _ in range(n_days)]
})
fluxes.to_csv('results/fluxes.csv', index=False)

# Plot results
plt.figure(figsize=(12, 6))
for i, name in enumerate(pool_names):
    plt.plot(sol.t, sol.y[i], label=name)
plt.xlabel('Time (days)')
plt.ylabel('C (g m⁻²)')
plt.legend()
plt.savefig('results/carbon_balance.png')