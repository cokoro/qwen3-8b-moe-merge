import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# Constants
RUNOFF_DECAY = 2.0
K_BASEFLOW = 5e-3  # mm/s
SPECIFIC_YIELD = 0.20
THETA_RESIDUAL = 0.02
DT_SECONDS = 3600

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def read_csv(path):
    return pd.read_csv(path)

def write_csv(df, path):
    df.to_csv(path, index=False)

def write_npy(arr, path):
    np.save(path, arr)

def plot_water_balance(balances, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(balances['cumulative'], label='Cumulative')
    ax1.legend()
    ax2.plot(balances['moisture_profile'], label='Mid-depth')
    ax2.legend()
    plt.savefig(filename)
    plt.close()

def analysis():
    # Read inputs
    soil = read_json('data/soil_profile.json')
    forcing = read_csv('data/forcing.csv')
    site = read_json('data/site.json')
    
    # Setup
    n_days = site['n_days']
    dt = site['dt_seconds']
    topo_idx = site['topo_index']
    init_water_table = site['init_water_table_m']
    n_layers = soil['n_layers']
    layer_thick = np.array(soil['layer_thicknesses_m'])
    layer_centers = np.array(soil['layer_centers_m'])
    theta_sat = np.array(soil['theta_sat'])
    psi_sat = np.array(soil['psi_sat_m'])
    b = np.array(soil['b_param'])
    k_sat = np.array(soil['k_sat_m_s'])
    theta_init = np.array(soil['theta_init'])
    
    # Precompute layer properties
    layer_depths = np.concatenate([[0], layer_thick.cumsum()])
    layer_depths = layer_depths[1:]  # bottom layer is at depth = sum(thick)
    layer_depths = layer_depths - layer_centers  # depth from bottom to center
    layer_depths = layer_depths[::-1]  # reverse for top-down indexing
    
    # Initialize state
    theta = theta_init.copy()
    water_table_depth = init_water_table
    aquifer_storage = 0.0
    moisture_profile = theta.copy()
    fluxes = np.zeros((n_days, 6))
    balances = {'cumulative': [], 'moisture_profile': []}
    
    # Process each day
    for day in range(n_days):
        precip = forcing.iloc[day]['precip_mm_day']
        pet = forcing.iloc[day]['pet_mm_day']
        day_start = day * dt
        
        # Compute surface runoff
        sat_frac = 0.5 * np.exp(-RUNOFF_DECAY * water_table_depth) * np.exp(topo_idx / 10.0)
        sat_frac = np.clip(sat_frac, 0, 1)
        runoff_surface = sat_frac * precip
        infil = (1 - sat_frac) * precip
        
        # Compute ET
        root_frac = np.array([0.4, 0.3, 0.2, 0.1])  # root distribution
        theta_wilt = np.array([0.15, 0.15, 0.15, 0.15])  # theta at -150 m
        theta_fc = np.array([0.25, 0.25, 0.25, 0.25])  # theta at -3.3 m
        beta = np.sum((theta - theta_wilt) / (theta_fc - theta_wilt) * root_frac)
        beta = np.clip(beta, 0, 1)
        et_actual = beta * pet
        
        # Solve Richards equation
        # ... (tridiagonal solve implementation here)
        
        # Compute baseflow and recharge
        # ... (water table dynamics and baseflow calculation here)
        
        # Update storage and balances
        # ... (update aquifer storage and balances)
        
        # Record outputs
        fluxes[day, :] = [infil, runoff_surface, recharge, baseflow, drainage, et_actual]
        balances['cumulative'].append(sum(fluxes[:day+1, :]))
        balances['moisture_profile'].append(moisture_profile)
    
    # Write outputs
    write_npy(moisture_profile, 'results/soil_moisture.npy')
    write_csv(pd.DataFrame(balances), 'results/water_balance.csv')
    plot_water_balance(balances, 'results/water_balance.png')

if __name__ == '__main__':
    analysis()