import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from datetime import datetime, timedelta
import json

# Load input data
soil_props = json.load(open("data/soil_properties.json"))
climate = pd.read_csv("data/climate_forcing.csv")
litter = pd.read_csv("data/litter_input.csv")

# Initialize soil properties
pool_names = soil_props["pool_names"]
C_pool_ratios = soil_props["C:N_ratios"]
clay_fraction = soil_props["clay_fraction"]
Q10 = soil_props["Q10"]
T_ref = 20
field_capacity = soil_props["field_capacity"]
wilting_point = soil_props["wilting_point"]
initial_C = soil_props["initial_C"]
initial_N = soil_props["initial_N"]

# Initialize state vector
state = np.array([initial_C[i] / C_pool_ratios[i] for i in range(len(C_pool_ratios))])  # N pools
state = np.concatenate((state, state * np.array(C_pool_ratios)))  # C pools

# Define ODE system
def model(t, state):
    C_pools = state[:len(C_pool_ratios)]
    N_pools = state[len(C_pool_ratios):]
    
    # Calculate environmental scalars
    temp_scalar = Q10 ** ((climate[climate['day'] == t]['temperature_C'].values[0] - T_ref) / 10)
    moisture_scalar = (climate[climate['day'] == t]['soil_moisture_m3m3'].values[0] - wilting_point) / (field_capacity - wilting_point)
    
    # Decomposition rates
    rates = np.array([14.8, 3.9, 7.3 * (1 - clay_fraction), 0.20, 0.0045])
    rates *= temp_scalar * moisture_scalar
    
    # Litter input
    litter_C = litter[litter['day'] == t]['litter_c_gCm2day'].values[0]
    litter_N = litter[litter['day'] == t]['litter_n_gNm2day'].values[0]
    litter_C = litter_C / C_pool_ratios[0]  # Convert to N pool
    litter_N = litter_N / C_pool_ratios[0]  # Convert to N pool
    
    # Transfer between pools
    transfers = np.zeros_like(C_pools)
    transfers[0] = rates[0] * C_pools[0] * 0.55  # METL to Active
    transfers[1] = rates[1] * C_pools[1] * 0.55  # STRL to Active
    transfers[2] = rates[2] * C_pools[2] * 0.55  # Active to Slow
    transfers[3] = rates[3] * C_pools[3] * 0.55  # Slow to Active
    transfers[4] = rates[4] * C_pools[4] * 0.55  # Passive to Active
    
    # Clay stabilization
    transfer_eff = 1 - (0.85 - 0.68 * clay_fraction)
    transfers[2] = transfer_eff * transfers[2]
    transfers[4] = (1 - transfer_eff) * transfers[2]
    
    # Update C pools
    C_pools = C_pools - transfers + np.concatenate((litter_C, np.zeros(4)))
    
    # Calculate N transfers
    N_transfers = transfers * (np.array(C_pool_ratios) / C_pool_ratios[0])
    
    # Calculate N mineralization
    N_mineralization = np.sum(N_transfers) - N_pools[0] * (np.array(C_pool_ratios) / C_pool_ratios[0])
    
    # Update N pools
    N_pools = N_pools - N_transfers + N_mineralization
    
    # Return derivatives
    return np.concatenate((C_pools, N_pools))

# Solve ODE
t_eval = np.arange(0, 365 * 10, 1)  # 10 years
solution = solve_ivp(model, (0, t_eval[-1]), state, t_eval=t_eval, method='BDF', rtol=1e-6, atol=1e-8)

# Save results
os.makedirs("results", exist_ok=True)
np.save("results/carbon_pools.npy", solution.y[:5].T)
np.save("results/nitrogen_pools.npy", solution.y[5:].T)
fluxes = pd.DataFrame({
    'day': t_eval,
    'soil_respiration_gC': solution.y[:5].T[0] * C_pool_ratios[0],
    'net_n_mineralization_gN': solution.y[5:].T[0] * C_pool_ratios[0],
    'n_immobilization_gN': solution.y[5:].T[0] * C_pool_ratios[0] - solution.y[5:].T[0] * C_pool_ratios[0]
})
fluxes.to_csv("results/fluxes.csv", index=False)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(solution.t, solution.y[:5].T)
plt.xlabel("Time (days)")
plt.ylabel("Carbon pools (g C m-2)")
plt.title("Carbon pool evolution over time")
plt.legend(pool_names)
plt.savefig("results/carbon_balance.png")
plt.close()