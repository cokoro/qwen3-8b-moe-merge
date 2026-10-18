import numpy as np
import matplotlib.pyplot as plt
import os
import json

# Load input data
with open('data/soil_profile.json') as f:
    soil = json.load(f)
n_layers = soil['n_layers']
layer_thicknesses_m = np.array(soil['layer_thicknesses_m'])
layer_centers_m = np.array(soil['layer_centers_m'])
theta_sat = np.array(soil['theta_sat'])
psi_sat_m = np.array(soil['psi_sat_m'])
b_param = np.array(soil['b_param'])
k_sat_m_s = np.array(soil['k_sat_m_s'])
theta_init = np.array(soil['theta_init'])

with open('data/forcing.csv') as f:
    forcing = np.loadtxt(f,  delimiter=',', skiprows=1)
precip_mm_day = forcing[:, 1]
pet_mm_day = forcing[:, 2]

with open('data/site.json') as f:
    site = json.load(f)
topo_index = site['topo_index']
init_water_table_m = site['init_water_table_m']
dt_seconds = site['dt_seconds']
n_days = site['n_days']

RUNOFF_DECAY = 2.0
K_baseflow_mm_s = 5e-3
specific_yield = 0.20
theta_residual = 0.02

# Precompute layer properties
layer_thickness