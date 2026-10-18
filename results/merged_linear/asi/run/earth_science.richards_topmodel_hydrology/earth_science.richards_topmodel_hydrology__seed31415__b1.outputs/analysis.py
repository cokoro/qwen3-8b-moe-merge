import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
RUNOFF_DECAY = 2.0
K_BASEFLOW_MM_S = 5e-3
SPECIFIC_YIELD = 0.20
THETA_RESIDUAL = 0.02

def load_data():
    soil_profile = np.load("data/soil_profile.json")
    forcing = pd.read_csv("data/forcing.csv")
    site = np.load("data/site.json")
    return soil_profile, forcing, site

def compute_hydraulic_properties(theta, layer):
    S = np.clip(theta / layer['theta_sat'], 0.01, 1.0)
    psi = layer['psi_sat_m'] * S ** (-layer['b_param'])
    K = layer['k_sat_m_s'] * S ** (2 * layer['b_param'] + 3)
    D = -K * layer['b_param'] * psi / theta
    return S, psi, K, D

def solve_richards(theta, layer, dt, K, D, infiltration, et_actual, psi_bottom, water_table_depth):
    # Construct tridiagonal matrix
    diag = np.zeros((layer['n_layers'], 1))
    off_diag = np.zeros((layer['n_layers'], 1))
    rhs = np.zeros((layer['n_layers'], 1))
    # Build the matrix
    for i in range(layer['n_layers']):
        diag[i, 0] = 1 / dt + K[i] / layer['layer_thicknesses_m'][i]
        if i > 0:
            off_diag[i-1, 0] = -K[i-1] / layer['layer_thicknesses_m'][i-1]
        if i < layer['n_layers'] - 1:
            off_diag[i, 0] = -K[i] / layer['layer_thicknesses_m'][i]
        # Right-hand side
        rhs[i, 0] = theta[i] / dt - et_actual[i] / layer['layer_thicknesses_m'][i] + infiltration
    # Add bottom drainage
    if water_table_depth > 0:
        if water_table_depth > layer['layer_thicknesses_m'][-1] + layer['layer_centers_m'][-1]:
            # Water table below bottom layer
            drainage_bot = 0.0
        else:
            # Water table above bottom layer
            depth = water_table_depth - layer['layer_centers_m'][-1]
            recharge = K[-1] * (0.0 - psi_bottom) / depth + K[-1]
            recharge = np.clip(recharge, -2e-6, 2e-6)
            drainage_bot = -recharge
        rhs[-1, 0] += drainage_bot / layer['layer_thicknesses_m'][-1]
    # Solve tridiagonal system
    banded = np.vstack([off_diag, diag, off_diag])
    theta_new = np.linalg.solve_banded((1, 1), banded, rhs)
    return theta_new

def main():
    soil_profile, forcing, site = load_data()
    n_layers = soil_profile['n_layers']
    layer_thicknesses = soil_profile['layer_thicknesses_m']
    layer_centers = soil_profile['layer_centers_m']
    theta_sat = soil_profile['theta_sat']
    psi_sat = soil_profile['psi_sat_m']
    b_param = soil_profile['b_param']
    k_sat = soil_profile['k_sat_m_s']
    theta_init = soil_profile['theta_init']
    topo_index = site['topo_index']
    init_water_table = site['init_water_table_m']
    dt_seconds = site['dt_seconds']
    n_days = site['n_days']
    
    # Initialize state
    theta = theta_init.copy()
    water_table_depth = init_water_table
    aquifer_storage = 0.0
    infiltration = np.zeros(n_days)
    runoff_surface = np.zeros(n_days)
    recharge = np.zeros(n_days)
    baseflow = np.zeros(n_days)
    drainage_bot = np.zeros(n_days)
    et_actual = np.zeros(n_days)
    day = 0
    
    # Run simulation
    for day in range(n_days):
        precip = forcing.loc[day, 'precip_mm_day']
        pet = forcing.loc[day, 'pet_mm_day']
        # Compute surface runoff
        sat_frac = 0.5 * np.exp(-RUNOFF_DECAY * water_table_depth) * np.exp(topo_index / 10.0)
        sat_frac = np.clip(sat_frac, 0, 1)
        runoff_surface[day] = sat_frac * precip
        infiltration[day] = (1 - sat_frac) * precip
        # Compute evapotranspiration
        root_frac = np.array([0.4, 0.3, 0.2, 0.1])
        theta_wilt = theta_sat.copy() * np.exp(-150.0 / psi_sat)
        theta_fc = theta_sat.copy() * np.exp(3.3 / psi_sat)
        beta = np.sum((theta - theta_wilt) / (theta_fc - theta_wilt) * root_frac)
        beta = np.clip(beta, 0, 1)
        et_actual[day] = beta * pet
        # Compute hydraulic properties
        S = np.zeros(n_layers)
        psi = np.zeros(n_layers)
        K = np.zeros(n_layers)
        D = np.zeros(n_layers)
        for i in range(n_layers):
            S[i], psi[i], K[i], D[i] = compute_hydraulic_properties(theta[i], {
                'theta_sat': theta_sat[i],
                'psi_sat_m': psi_sat[i],
                'b_param': b_param[i],
                'k_sat_m_s': k_sat[i],
                'layer_thicknesses_m': layer_thicknesses[i],
                'layer_centers_m': layer_centers[i]
            })
        # Solve Richards equation
        theta_new = solve_richards(theta, {
            'n_layers': n_layers,
            'layer_thicknesses_m': layer_thicknesses,
            'layer_centers_m': layer_centers,
            'theta_sat': theta_sat,
            'psi_sat_m': psi_sat,
            'b_param': b_param,
            'k_sat_m_s': k_sat,
            'layer_centers_m': layer_centers
        }, dt_seconds, K, D, infiltration[day], et_actual[day], psi[-1], water_table_depth)
        theta = theta_new
        # Compute water table dynamics
        if water_table_depth > 0:
            if water_table_depth > layer_thicknesses[-1] + layer_centers[-1]:
                # Water table below bottom layer
                drainage_bot[day] = 0.0
            else:
                # Water table above bottom layer
                depth = water_table_depth - layer_centers[-1]
                recharge = K[-1] * (0.0 - psi[-1]) / depth + K[-1]
                recharge = np.clip(recharge, -2e-6, 2e-6)
                drainage_bot[day] = -recharge
        else:
            drainage_bot[day] = 0.0
        # Compute baseflow
        baseflow[day] = K_BASEFLOW_MM_S * np.exp(-topo_index) * np.exp(-RUNOFF_DECAY * water_table_depth)
        # Update water table depth
        if water_table_depth > 0:
            net_flux_mm = (recharge - baseflow[day]) * dt_seconds / 3600.0 * 1000.0
            aquifer_storage += net_flux_mm
            water_table_depth = water_table_depth - net_flux_mm / SPECIFIC_YIELD / 1000.0
        else:
            water_table_depth = max(water_table_depth, -layer_thicknesses[-1] * 10.0)
        # Clip negative theta values
        theta = np.clip(theta, THETA_RESIDUAL, theta_sat)
        # Save outputs
        np.save("results/soil_moisture.npy", theta)
        pd.DataFrame({
            'day': [day + 1],
            'water_table_m': [water_table_depth],
            'aquifer_storage_mm': [aquifer_storage]
        }).to_csv("results/water_table.csv", index=False, mode='a', header=False)
        pd.DataFrame({
            'day': [day + 1],
            'infiltration': [infiltration[day]],
            'runoff_surface': [runoff_surface[day]],
            'recharge': [recharge],
            'baseflow': [baseflow[day]],
            'drainage_bot': [drainage_bot[day]],
            'et_actual': [et_actual[day]]
        }).to_csv("results/fluxes.csv", index=False, mode='a', header=False)
    # Plot water balance
    plt.figure(figsize=(12, 6))
    plt.plot(forcing['day'], forcing['precip_mm_day'], label='Precipitation')
    plt.plot(forcing['day'], et_actual, label='ET Actual')
    plt.plot(forcing['day'], runoff_surface, label='Runoff')
    plt.plot(forcing['day'], baseflow, label='Baseflow')
    plt.plot(forcing['day'], drainage_bot, label='Drainage')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Water balance (mm/day)')
    plt.savefig('results/water_balance.png')
    plt.close()

if __name__ == "__main__":
    main()