```python
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Read input data
with open('data/soil_profile.json') as f:
    soil_profile = json.load(f)
n_layers = soil_profile['n_layers']
layer_thicknesses = np.array(soil_profile['layer_thicknesses_m'])
layer_centers = np.array(soil_profile['layer_centers_m'])
theta_sat = np.array(soil_profile['theta_sat'])
psi_sat = np.array(soil_profile['psi_sat_m'])
b_param = np.array(soil_profile['b_param'])
k_sat = np.array(soil_profile['k_sat_m_s'])
theta_init = np.array(soil_profile['theta_init'])

with open('data/site.json') as f:
    site = json.load(f)
topo
```