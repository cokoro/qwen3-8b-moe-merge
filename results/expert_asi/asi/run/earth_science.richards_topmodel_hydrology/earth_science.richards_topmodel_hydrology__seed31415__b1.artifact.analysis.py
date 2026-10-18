import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os

def analysis():
    # Load data
    with open('data/site.json') as f:
        site = json.load(f)
    with open('data/soil_profile.json') as f:
        soil = json.load(f)
    df = pd.read_csv('data/forcing.csv')
    
    n_layers = soil['n_layers']
    layer_centers = np.array(soil['layer_centers_m'])
    layer_thicknesses = np.array(soil['layer_thicknesses_m'])
    layer_thicknesses_cum = np.r_[0, np.cumsum(layer_thicknesses)]
    
    # Constants
    RUNOFF_DECAY = 2.0
    K_baseflow = 5e-3  # mm/s
    specific_yield = 0.20
    theta_residual = 0.02
    
    # Initialize state variables
    theta = np.array(soil['theta_init'], dtype=np.float64)
    water_table_depth = site['init_water_table_m']
    aquifer_storage = 0.0
    storage = np.sum(theta * layer_thicknesses) * 1e3  # mm
    
    # Time step
    dt_sec = site['dt_seconds']
    n_days = site['n_days']
    days = np.arange(n_days)
    
    # Output arrays
    soil_moisture = np.zeros((n_days, n_layers))
    water_table = np.zeros((n_days, 2))
    fluxes = np.zeros((n_days, 6))
    
    # Root distribution
    root_frac = np.array([0.4, 0.3, 0.2, 0.1])
    
    # Hydraulic parameters
    for l in range(n_layers):
        K = soil['k_sat_m_s'][l]
        b = soil['b_param'][l]
        theta_sat = soil['theta_sat'][l]
        psi_sat = soil['psi_sat_m'][l]
        theta_wilt = theta_sat * (np.exp(-150.0 / psi_sat) if psi_sat < 0 else 0.0)
        theta_fc = theta_sat * (np.exp(-3.3 / psi_sat) if psi_sat < 0 else 0.0)
    
    # Main time loop
    for day in range(n_days):
        precip = df.iloc[day].precip_mm_day
        pet = df.iloc[day].pet_mm_day
        
        # Surface runoff
        sat_frac = 0.5 * np.exp(-RUNOFF_DECAY * water_table_depth) * np.exp(site['topo_index'] / 10.0)
        sat_frac = np.clip(sat_frac, 0, 1)
        runoff_surface = sat_frac * precip
        infil = (1 - sat_frac) * precip
        
        # Evapotranspiration
        beta = 0.0
        for l in range(n_layers):
            theta_l = theta[l]
            if theta_l < theta_wilt:
                beta += 0.0
            else:
                beta += (theta_l - theta_wilt) / (theta_fc - theta_wilt) * root_frac[l]
        beta = np.clip(beta, 0, 1)
        et_actual = beta * pet
        
        # Richards equation (implicit)
        # ... (full implementation omitted for brevity)
        
        # Water-table dynamics
        # ... (full implementation omitted for brevity)
        
        # Mass balance
        # ... (full implementation omitted for brevity)
        
        # Save outputs
        soil_moisture[day] = theta.copy()
        water_table[day] = [water_table_depth, aquifer_storage]
        fluxes[day] = [infil, runoff_surface, recharge, baseflow, drainage_bot, et_actual]
    
    # Plot water balance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(df['day'], df['precip_mm_day'] - df['pet_mm_day'] - df['runoff_surface'] - df['baseflow'] - df['drainage_bot'], 'r-')
    ax1.axhline(0, color='k', linestyle='--')
    ax2.plot(layer_centers, theta, 'o-', label='Topsoil')
    ax2.set_xlabel('Depth (m)')
    ax2.set_ylabel('Volumetric moisture (m^3/m^3)')
    plt.savefig('results/water_balance.png')
    np.save('results/soil_moisture.npy', soil_moisture)
    pd.DataFrame(water_table, columns=['day', 'water_table_m', 'aquifer_storage_mm']).to_csv('results/water_table.csv', index=False)
    pd.DataFrame(fluxes, columns=['day', 'infiltration', 'runoff_surface', 'recharge', 'baseflow', 'drainage_bot', 'et_actual']).to_csv('results/fluxes.csv', index=False)