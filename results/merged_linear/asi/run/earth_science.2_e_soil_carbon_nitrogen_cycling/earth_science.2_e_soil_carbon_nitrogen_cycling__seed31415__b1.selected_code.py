import numpy as np
import pandas as pd
import os
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load soil properties
with open('data/soil_properties.json') as f:
    soil_props = json.load(f)

# Initialize simulation parameters
n_days = 365  # Simulate one year for simplicity; change to 365*20 for full period
dt_days = 1  # Daily time steps
t_span = (0, n_days)
t_eval = np.arange(0, n_days + 1)
initial_C = np.array(soil_props['initial_C_pool_sizes'])  # g C m⁻²
initial_N = initial_C * (soil_props['C_N_ratios'] / 14.006)  # Convert to g N m⁻² using C=12.01

# Climate data
climate_df = pd.read_csv('data/climate_forcing.csv')
climate_interp = interp1d(climate_df['day'], climate_df['soil_moisture_m3m3'], kind='linear',
                          fill_value='extrapolate')

# Litter input
litter_df = pd.read_csv('data/litter_input.csv')
litter_interp = interp1d(litter_df['day'], litter_df[['litter_c_gCm2day', 'litter_n_gNm2day']],
                         kind='linear', fill_value='extrapolate')

# Q10 response function
Q10 = soil_props['Q10']
T_ref = 20  # °C

# Soil moisture response function
def moisture_scalar(moisture):
    return (moisture - soil_props['wilting_point']) / (soil_props['field_capacity'] - soil_props['wilting_point'])

# Carbon pool decomposition rates
base_rates = np.array([14.8, 3.9, 7.3, 0.20, 0.0045])  # yr⁻¹
temperature_scalar = lambda t: Q10 ** ((climate_df.loc[climate_df['day'] == t, 'temperature_C'].values[0] - T_ref) / 10)
moisture_scalar = lambda t: moisture_scalar(climate_interp(t))

# Pool-specific C:N ratios
C_N_ratios = np.array(soil_props['C_N_ratios'])

# Clay stabilization coefficient
clay_fraction = soil_props['clay_fraction']
active_to_slow_transfer = 1 - (0.85 - 0.68 * clay_fraction)

# C-N stoichiometric coupling
def calculate_mineralization(N_pool, C_pool, C_N_ratio, C_pool_dest):
    required_N = C_pool_dest * C_N_ratio / 12.01  # g N m⁻²
    mineralization = max(N_pool - required_N, 0)
    return mineralization

# ODE system
def dCdt(t, y):
    C = y[:5]
    N = y[5:]
    dCdt = np.zeros_like(C)
    
    # METL decomposition
    metl_rate = base_rates[0] * temperature_scalar(t) * moisture_scalar(t)
    metl_respiration = metl_rate * C[0]
    dCdt[0] = -metl_respiration
    
    # STRL decomposition
    strl_rate = base_rates[1] * temperature_scalar(t) * moisture_scalar(t)
    strl_respiration = strl_rate * C[1]
    dCdt[1] = -strl_respiration
    
    # Active SOM decomposition
    active_rate = base_rates[2] * temperature_scalar(t) * moisture_scalar(t)
    active_respiration = active_rate * C[2]
    dCdt[2] = -active_respiration
    
    # Slow SOM decomposition
    slow_rate = base_rates[3] * temperature_scalar(t) * moisture_scalar(t)
    slow_respiration = slow_rate * C[3]
    dCdt[3] = -slow_respiration
    
    # Passive SOM decomposition
    passive_rate = base_rates[4] * temperature_scalar(t) * moisture_scalar(t)
    passive_respiration = passive_rate * C[4]
    dCdt[4] = -passive_respiration
    
    return dCdt

# Solve ODEs
sol = solve_ivp(dCdt, t_span, initial_C, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)

# Calculate daily fluxes
fluxes = np.zeros((n_days, 3))
for i in range(n_days):
    t = i
    C = sol.y[:, i]
    N = initial_N + calculate_mineralization(N_pool=initial_N, C_pool=C, C_N_ratio=C_N_ratios, C_pool_dest=C)
    fluxes[i, 0] = sol.y[0, i] * 1000  # Convert to g C m⁻² day⁻¹
    fluxes[i, 1] = calculate_mineralization(N_pool=initial_N, C_pool=C, C_N_ratio=C_N_ratios, C_pool_dest=C)
    fluxes[i, 2] = fluxes[i, 1] - fluxes[i, 0]

# Save results
os.makedirs('results', exist_ok=True)
np.save('results/carbon_pools.npy', sol.y.T)
np.save('results/nitrogen_pools.npy', initial_N)
pd.DataFrame(fluxes, columns=['soil_respiration_gC', 'net_n_mineralization_gN', 'n_immobilization_gN']).to_csv('results/fluxes.csv', index=False)

# Plot carbon balance
plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y.T, label=['Metabolic Litter', 'Structural Litter', 'Active SOM', 'Slow SOM', 'Passive SOM'])
plt.plot(sol.t, np.sum(sol.y, axis=1), label='Total SOC')
plt.xlabel('Time (days)')
plt.ylabel('Carbon (g C m⁻²)')
plt.title('Carbon Pool Evolution Over Time')
plt.legend()
plt.savefig('results/carbon_balance.png')
plt.close()