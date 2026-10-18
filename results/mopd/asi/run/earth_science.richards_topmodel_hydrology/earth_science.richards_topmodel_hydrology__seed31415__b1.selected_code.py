import numpy as np
import json
import matplotlib.pyplot as plt
import os
import pandas as pd

RUNOFF_DECAY = 2.0
K_baseflow = 5e-3
specific_yield = 0.20
theta_residual = 0.02
theta_wilt = 0.05
theta_fc = 0.25
root_frac = [0.4, 0.3, 0.2, 0.1]
theta_at_psi = lambda psi: theta_sat * (1 - np.exp(-psi / (b_param * 3.3)))
theta_at_psi_neg150 = theta_at_psi(-150)
theta_at_psi_neg3_3 = theta_at_psi(-3.3)

def analysis():
    with open('data/soil_profile.json') as f:
        profile = json.load(f)
    n_layers = profile['n_layers']
    layer_thicknesses = np.array(profile['layer_thicknesses_m'])
    layer_centers = np.array(profile['layer_centers_m'])
    theta_sat = np.array(profile['theta_sat'])
    psi_sat = np.array(profile['psi_sat_m'])
    b_param = np.array(profile['b_param'])
    k_sat = np.array(profile['k_sat_m_s'])
    theta_init = np.array(profile['theta_init'])
    
    with open('data/forcing.csv') as f:
        df = pd.read_csv(f, index_col=0)
    days = df.index.values
    precip = df['precip_mm_day'].values
    pet = df['pet_mm_day'].values
    
    with open('data/site.json') as f:
        site = json.load(f)
    topo_index = site['topo_index']
    init_water_table_m = site['init_water_table_m']
    dt_seconds = site['dt_seconds']
    n_days = site['n_days']
    
    theta = theta_init.copy()
    water_table_depth = init_water_table_m
    aquifer_storage = 0.0
    fluxes = np.zeros((n_days, 6))
    water_table = np.zeros(n_days)
    soil_moisture = np.zeros((n_days, n_layers))
    
    for day in range(n_days):
        dt = dt_seconds
        if day > 0:
            dt = dt_seconds * 3600 / (days[day] - days[day-1])
        
        # Surface runoff
        sat_frac = 0.5 * np.exp(-RUNOFF_DECAY * water_table_depth) * np.exp(topo_index / 10.0)
        sat_frac = np.clip(sat_frac, 0, 1)
        runoff_surface = sat_frac * precip[day]
        infiltration = (1 - sat_frac) * precip[day]
        
        # ET
        theta_wilt_layer = theta_at_psi_neg150
        theta_fc_layer = theta_at_psi_neg3_3
        beta = np.zeros(n_layers)
        for l in range(n_layers):
            theta_l = theta[l]
            theta_wilt = theta_wilt_layer[l]
            theta_fc = theta_fc_layer[l]
            beta[l] = np.clip((theta_l - theta_wilt) / (theta_fc - theta_wilt) * root_frac[l], 0, 1)
        et_actual = beta * pet[day]
        
        # Richards equation
        D = np.zeros(n_layers)
        K = np.zeros(n_layers)
        for l in range(n_layers):
            S = max(theta[l] / theta_sat[l], 0.01)
            psi = psi_sat[l] * S**(-b_param[l])
            K[l] = k_sat[l] * S**(2*b_param[l] + 3)
            D[l] = -K[l] * b_param[l] * psi / theta[l]
        
        # Bottom drainage
        K_bottom = K[-1]
        psi_bottom = psi_sat[-1] * (theta[-1] / theta_sat[-1])**(-b_param[-1])
        psi_table = 0
        z_bottom = layer_centers[-1]
        recharge = -K_bottom * (psi_table - psi_bottom) / (water_table_depth - z_bottom) + K_bottom
        recharge = np.clip(recharge, -2e-6, 2e-6)
        drainage_bot = recharge
        
        # Tridiagonal matrix
        diag = np.zeros(n_layers)
        off_diag = np.zeros(n_layers-1)
        for l in range(n_layers):
            diag[l] = 1 / dt + (K[l] + K[l+1] if l < n_layers-1 else K[l]) / layer_thicknesses[l]
            if l < n_layers-1:
                off_diag[l] = -K[l+1] / layer_thicknesses[l]
        rhs = np.zeros(n_layers)
        rhs[0] = infiltration / dt
        rhs[1:] = -et_actual[1:] / dt
        
        # Thomas algorithm
        for l in range(n_layers):
            if l == 0:
                diag[l] /= 1
                rhs[l] /= 1
            else:
                off_diag[l-1] /= diag[l-1]
                rhs[l] -= off_diag[l-1] * rhs[l-1]
                diag[l] -= off_diag[l-1] * off_diag[l-1]
                rhs[l] /= diag[l]
        theta_new = np.zeros(n_layers)
        theta_new[-1] = rhs[-1]
        for l in range(n_layers-1, 0, -1):
            theta_new[l-1] = rhs[l-1] - off_diag[l-1] * theta_new[l]
        
        # Adjust theta
        theta = np.clip(theta_new, theta_residual, theta_sat)
        
        # Water table update
        net_flux_mm = (recharge - baseflow) * dt * 1000
        aquifer_storage += net_flux_mm
        water_table_depth = np.clip(water_table_depth - net_flux_mm / specific_yield / 1000, 0, 5)
        water_table[day] = water_table_depth
        
        # Save results
        fluxes[day, 0] = infiltration
        fluxes[day, 1] = runoff_surface
        fluxes[day, 2] = recharge
        fluxes[day, 3] = baseflow
        fluxes[day, 4] = drainage_bot
        fluxes[day, 5] = et_actual
        soil_moisture[day] = theta
    
    # Save outputs
    np.save('results/soil_moisture.npy', soil_moisture)
    pd.DataFrame({
        'day': days,
        'water_table_m': water_table,
        'aquifer_storage_mm': aquifer_storage
    }).to_csv('results/water_table.csv', index=False)
    pd.DataFrame({
        'day': days,
        'infiltration': fluxes[:,0],
        'runoff_surface': fluxes[:,1],
        'recharge': fluxes[:,2],
        'baseflow': fluxes[:,3],
        'drainage_bot': fluxes[:,4],
        'et_actual': fluxes[:,5]
    }).to_csv('results/fluxes.csv', index=False)
    
    # Plot water balance
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(days, np.cumsum(precip), label='Precipitation')
    ax1.plot(days, np.cumsum(et_actual), label='ET')
    ax1.plot(days, np.cumsum(runoff_surface), label='Runoff')
    ax1.plot(days, np.cumsum(fluxes[:,2] + fluxes[:,3]), label='Recharge & Baseflow')
    ax1.plot(days, np.cumsum(fluxes[:,4]), label='Drainage')
    ax1.legend()
    ax1.set_title('Cumulative Water Balance')
    
    ax2.plot(days, soil_moisture[:,1], label='Mid-depth')
    ax2.set_title('Soil Moisture Profile')
    plt.tight_layout()
    plt.savefig('results/water_balance.png')