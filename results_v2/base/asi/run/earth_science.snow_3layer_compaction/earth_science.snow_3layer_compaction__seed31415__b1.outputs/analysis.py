import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta

# Constants
T_FREEZE = 273.15  # K
RHO_ICE = 917  # kg/m^3
C_ICE = 2114  # J/kg/K
L_F = 334000  # J/kg
SIGMA = 5.67e-8  # W/m^2/K^4
RHO_AIR = 1.225  # kg/m^3
C_P = 1005  # J/kg/K
R_A = 50  # s/m
K_EFF = lambda rho: 2.22e-2 + 7.75e-7 * rho**2
ALPHA_DRY = 0.80
ALPHA_WET = 0.55
THETA_IRR = 0.05
DT = 3600  # 1 hour
DT_SURF = 60  # 1 minute for sub-steps

# Parameters
C1_META = 2.777e-6  # 1/s
C2_META = 0.04  # 1/K
ETA0 = 7.62e6  # Pa s
C5 = 0.08  # 1/K
C6 = 0.023  # m^3/kg
RHO_DEST_THRESH = 150  # kg/m^3
RHO_NEW_MIN = 50
RHO_NEW_MAX = 200

def load_data():
    with open('data/snowpack_init.json', 'r') as f:
        snowpack_init = json.load(f)
    forcing = pd.read_csv('data/forcing.csv', parse_dates=['time_hr'])
    return snowpack_init, forcing

def compute_density