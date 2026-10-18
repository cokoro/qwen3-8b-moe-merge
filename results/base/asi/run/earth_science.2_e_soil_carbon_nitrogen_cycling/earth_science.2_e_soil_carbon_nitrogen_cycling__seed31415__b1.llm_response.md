```python
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

# Define paths
data_dir = 'data'
results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

# Load input data
with open(os.path.join(data_dir, 'soil_properties.json')) as f:
    soil_props = json.load(f)
    pool_names = soil_props['pool_names']
    clay_fraction = soil_props['clay_fraction']
    q10 = soil_props['q10']
    field_capacity = soil_props['field_capacity']
    wilting_point = soil_props['wilting_point']
    initial_c = np.array(soil_props['initial_c'])
    initial_n = np.array(soil_props['initial_n'])
    c_n_ratios = np.array(soil_props['c_n_ratios'])

climate_df = pd.read_csv(os.path.join(data_dir, 'climate_forcing.csv'))
litter_df = pd.read_csv(os.path.join(data_dir, 'litter_input.csv'))

# Convert to datetime
climate_df['day'] = pd.to_datetime(climate_df['day'])
litter_df['day'] = pd.to_datetime(litter_df['day'])

# Define parameters
base_decomposition_rates = np.array([14.8, 3.9, 7.3, 0.20, 0.0045])  # yr⁻¹
c_use_efficiency = 0.55
clay_stabilization_factor = 0.85 - 0.68 * clay_fraction
active_to_slow_transfer_efficiency = 1 - clay_stabilization_factor
active_to_passive_transfer_efficiency = 0.05  # small fraction

def get_moisture_scalar(soil_moisture):
    if soil_moisture <= wilting_point:
        return 0.0
    elif soil_moisture >= field_capacity:
        return 1.0
    else:
        return (soil_moisture - wilting_point) / (field_capacity - wilting_point)

def temperature_scalar(temperature):
    T_ref = 20.0
    return q10 ** ((temperature - T_ref) / 10)

def calculate_transfer_efficiencies(clay_fraction):
    active_to_slow_transfer_efficiency = 1 - (0.85 - 0.68 * clay_fraction)
    active_to_passive_transfer_efficiency = 0.05
    return active_to_slow_transfer_efficiency, active_to_passive_transfer_efficiency

def model(t, y):
    c_pools = y[:5]
    n_pools = y[5:]
    total_c = np.sum(c_pools)
    total_n = np.sum(n_pools)
    
    # Litter input
    day = t
    litter_day = climate_df[climate_df['day'] == day].iloc[0]
    litter_c = litter_day['litter_c_gCm2day']
    litter_n = litter_day['litter_n_gNm2day']
    
    # Litter partitioning
    litter_c_n_ratio = litter_c / litter_n
    metl_fraction = 1.0 / (1.0 + (litter_c_n_ratio / 15.0))
    strl_fraction = 1.0 - metl_fraction
    
    # Temperature and moisture scalars
    temp_scalar = temperature_scalar(litter_day['temperature_C'])
    moisture_scalar = get_moisture_scalar(litter_day['soil_moisture_m3m3'])
    combined_scalar = temp_scalar * moisture_scalar
    
    # Base decomposition rates
    decomposition_rates = base_decomposition_rates * combined_scalar
    
    # Decomposition and transfer
    c_losses = np.zeros(5)
    n_losses = np.zeros(5)
    c_transfers = np.zeros(5)
    n_transfers = np.zeros(5)
    
    # METL decomposition
    metl_loss = decomposition_rates[0] * c_pools[0]
    c_losses[0] = metl_loss
    n_losses[0] = metl_loss / c_n_ratios[0]
    
    # STRL decomposition
    strl_loss = decomposition_rates[1] * c_pools[1]
    c_losses[1] = strl_loss
    n_losses[1] = strl_loss / c_n_ratios[1]
    
    # Active SOM decomposition
    active_loss = decomposition_rates[2] * c_pools[2]
    c_losses[2] = active_loss
    n_losses[2] = active_loss / c_n_ratios[2]
    
    # Slow SOM decomposition
    slow_loss = decomposition_rates[3] * c_pools[3]
    c_losses[3] = slow_loss
    n_losses[3] = slow_loss / c_n_ratios[3]
    
    # Passive SOM decomposition
    passive_loss = decomposition_rates[4] * c_pools[4]
    c_losses[4] = passive_loss
    n_losses[4] = passive_loss / c_n_ratios[4]
    
    # Transfer from METL to Active
    metl_to_active = metl_loss * c_use_efficiency
    c_transfers[0] = metl_to_active
    n_transfers[0] = metl_to_active / c_n_ratios[0]
    
    # Transfer from STRL to Active and Slow
    strl_to_active = strl_loss * c_use_efficiency * 0.6
    strl_to_slow = strl_loss * c_use_efficiency * 0.4
    c_transfers[1] = strl_to_active
    n_transfers[1] = strl_to_active / c_n_ratios[1]
    c_transfers[3] = strl_to_slow
    n_transfers[3] = strl_to_slow / c_n_ratios[1]
    
    # Transfer from Active to Slow and Passive
    active_to_slow = active_loss * c_use_efficiency * active_to_slow_transfer_efficiency
    active_to_passive = active_loss * c_use_efficiency * active_to_passive_transfer_efficiency
    c_transfers[2] = active_to_slow
    n_transfers[2] = active_to_slow / c_n_ratios[2]
    c_transfers[4] = active_to_passive
    n_transfers[4] = active_to_passive / c_n_ratios[2]
    
    # Transfer from Slow to Active and Passive
    slow_to_active = slow_loss * c_use_efficiency * 0.5
    slow_to_passive = slow_loss * c_use_efficiency * 0.5
    c_transfers[3] = slow_to_active
    n_transfers[3] = slow_to_active / c_n_ratios[3]
    c_transfers[4] = slow_to_passive
    n_transfers[4] = slow_to_passive / c_n_ratios[3]
    
    # Transfer from Passive to Active
    passive_to_active = passive_loss * c_use_efficiency * 0.8
    c_transfers[4] = passive_to_active
    n_transfers[4] = passive_to_active / c_n_ratios[4]
    
    # Update pools
    new_c_pools = c_pools - c_losses + c_transfers
    new_n_pools = n_pools - n_losses + n_transfers
    
    # Net N mineralization and immobilization
    mineralization = np.sum(n_losses) - np.sum(n_transfers)
    if mineralization > 0:
        n_immobilization = 0.0
    else:
        n_immobilization = -mineralization
    
    # Return derivatives
    dc = new_c_pools - c_pools
    dn = new_n_pools - n_pools
    return np.concatenate((dc, dn))

# Initial conditions
initial_state = np.concatenate((initial_c, initial_n))

# Time span
start_date = climate_df['day'].iloc[0]
end_date = climate_df['day'].iloc[-1]
time_span = (end_date - start_date).days
t_span = (start_date, end_date)

# Solve ODE
sol = solve_ivp(model, t_span, initial_state, method='BDF', rtol=1e-6, atol=1e-8, t_eval=np.linspace(start_date, end_date, time_span + 1))

# Save results
carbon_pools = sol.y[:5].T
np.save(os.path.join(results_dir, 'carbon_pools.npy'), carbon_pools)
nitrogen_pools = sol.y[5:].T
np.save(os.path.join(results_dir, 'nitrogen_pools.npy'), nitrogen_pools)

# Calculate fluxes
fluxes = []
for i in range(time_span + 1):
    day = sol.t[i]
    litter_day = climate_df[climate_df['day'] == day].iloc[0]
    litter_c = litter_day['litter_c_gCm2day']
    litter_n = litter_day['litter_n_gNm2day']
    
    # Soil respiration
    respiration = np.sum(sol.y[:5, i] - sol.y[:5, i-1]) if i > 0 else 0
    
    # Net N mineralization
    mineralization = np.sum(sol.y[5:, i] - sol.y[5:, i-1]) if i > 0 else 0
    
    # N immobilization
    immobilization = -mineralization if mineralization < 0 else 0
    
    fluxes.append({
        'day': day,
        'soil_respiration_gC': respiration,
        'net_n_mineralization_gN': mineralization,
        'n_immobilization_gN': immobilization
    })

fluxes_df = pd.DataFrame(fluxes)
fluxes_df.to_csv(os.path.join(results_dir, 'fluxes.csv'), index=False)

# Plot carbon balance
plt.figure(figsize=(12, 6))
for i, pool in enumerate(pool_names):
    plt.plot(sol.t, sol.y[i], label=pool)
plt.plot(sol.t, np.sum(sol.y[:5], axis=0), label='Total SOC', linestyle='--', color='black')
plt.xlabel('Time (days)')
plt.ylabel('Carbon (g C m⁻²)')
plt.title('Carbon Pool Evolution Over Time')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'carbon_balance.png'))
plt.close()

# Verify mass balance
total_litter_c = np.sum(litter_df['litter_c_gCm2day'])
total_respiration = np.sum(sol.y[:5, -1] - sol.y[:5, 0])
mass_balance_error = abs((total_litter_c - total_respiration) / total_litter_c) * 100
print(f'Mass balance error: {mass_balance_error:.2f}%')

# Check non-negativity
non_negative = np.all(sol.y[:5] >= 0)
print(f'Non-negativity check: {non_negative}')

# Check turnover ordering
turnover_times = 1 / base_decomposition_rates
turnover_ordering = np.all(turnover_times[0] > turnover_times[1] > turnover_times[2] > turnover_times[3] > turnover_times[4])
print(f'Turnover ordering check: {turnover_ordering}')
```