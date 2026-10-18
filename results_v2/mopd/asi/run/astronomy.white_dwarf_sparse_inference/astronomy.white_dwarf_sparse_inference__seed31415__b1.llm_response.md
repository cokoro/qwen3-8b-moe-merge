```python
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import os
import sys
sys.path.append('..')
from physics import white_dwarf_ode, surface_event
from utils import load_data, save_data, plot_reconstruction

# Load data
mr_data = load_data('data/mr_observations.csv')
profile_data = load_data('data/profile_fragments.csv')
query_targets = json.load(open('data/query_targets.json'))
physics_manifest = json.load(open('data/physics_manifest.json'))

# Parse physics manifest
surface_density = physics_manifest['surface_density']
surface_density_unit = physics_manifest['surface_density_unit']
surface_rule = physics_manifest['surface_rule']
mass_unit = physics_manifest['mass_unit']
radius_unit = physics_manifest['radius_unit']
density_unit = physics_manifest['density_unit']
enclosed_mass_unit = physics_manifest['enclosed_mass_unit']
noise_std_dev = physics_manifest['noise_std_dev']

# Parse query targets
query_masses = [float(m) for m in query_targets['masses_solar']]
query_profiles = [float(m) for m in query_targets['profiles_masses_solar']]
query_profile_radius_fractions = query_targets['radius_fraction_grid']

# Build a dense dimensionless stable branch
def build_branch():
    # Use central density as the state variable
    central_density_grid = np.logspace(-6, 0, 1000)
    branch_data = []
    for central_density in central_density_grid:
        sol = solve_ivp(lambda r, y: white_dwarf_ode(r, y
```