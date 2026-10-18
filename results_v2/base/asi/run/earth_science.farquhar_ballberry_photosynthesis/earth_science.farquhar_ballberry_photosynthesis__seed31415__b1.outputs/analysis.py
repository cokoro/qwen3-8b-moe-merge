import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Load data
leaf_params = json.load(open('data/leaf_params.json'))
forcing_data = pd.read_csv('data/forcing.csv')
test_cases = json.load(open('data/test_cases.json'))

# Constants
R = 8.314  # J/mol/K
Q10_vcmax = 2.4
Kc25 = 30  # Pa
Ko25 = 30000  # Pa
Gamma_star = 0.5 * Kc / Ko * O2_pa * 0.21
Rd = 0.015 * Vcmax(T)
cf = p_atm / (R * T_air)
D_H2O_over_CO2 = 1.37
D_ratio = 1.65  # 1.37 * 1.20

# Temperature functions
def Vcmax(Tc):
    return Vcmax25 * Q10_vcmax**((Tc - 25)/10) / (1 + np.exp((-2.2e5 + 710*(Tc + 273.16))/(R*(Tc + 273.16))))

def Kc(Tc):
    return Kc25 * 2.1**((Tc - 25)/10)

def Ko(Tc):
    return Ko25 * 1.2**((Tc - 25)/10)

def Gamma_star_func(Tc, O2_pa):
    return 0.5 * Kc(Tc) / Ko(Tc) * O2_pa * 0.21

# Farquhar model
def compute_farquhar(APAR, Tc, Vcmax, Jmax, Rd, Kc, Ko, O2_pa, ci, pathway):
    if pathway == 'C3':
        # Compute J
        if Jmax is None:
            J = alpha_q * APAR
        else:
            # Non-rectangular hyperbola
            def J_func(J):
                return theta_J * J**2 - (alpha_q * APAR + Jmax) * J + alpha_q * APAR * Jmax
            J = root_scalar(J_func, bracket=[0, Jmax], method='brentq').root
        # Compute Wc, Wj, We
        Gamma_star = Gamma_star_func(Tc, O2_pa)
        Wc = Vcmax * (ci - Gamma_star) / (ci + Kc * (1 + O2_pa / Ko))
        Wj = J * (ci - Gamma_star) / (4 * ci + 8 * Gamma_star)
        We = 0.5 * Vcmax
        A = min(Wc, Wj, We) - Rd
    else:  # C4
        Wc = Vcmax
        Wj = alpha_q * APAR
        We = 4000 * Vcmax * ci / p_atm
        A = min(Wc, Wj, We) - Rd
    return A

# Ball-Berry model
def compute_ballberry(A_mol, cs, but not sure about the rest of the code is too long to fit in one code block. I'll need to continue the code in the next block.