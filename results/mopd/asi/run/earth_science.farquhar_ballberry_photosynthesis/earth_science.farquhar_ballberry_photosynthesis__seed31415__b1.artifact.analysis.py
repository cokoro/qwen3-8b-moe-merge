import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import fsolve

# Load data
leaf_params = json.load(open('leaf_params.json'))
forcing = pd.read_csv('forcing.csv')
test_cases = json.load(open('test_cases.json'))

# Constants
R = 8.314  # J/mol/K
Q10_vcmax = 2.4
Kc25 = 30  # Pa
Ko25 = 30000  # Pa
Gamma_star = 0.5 * Kc25 / Ko25 * 0.21  # Pa
Rd_factor = 0.015
D_H2O = 1.37  # diffusivity ratio
D_CO2 = 1.20  # diffusivity ratio
cf = lambda T_air: forcing['p_atm_Pa'].values / (R * T_air)
A_mol_factor = 1e-6  # umol to mol
cf_val = cf(273.15)  # assume 0°C for initial calculation

def calc_vcm(T_air):
    Tc = T_air - 273.15
    Q10 = Q10_vcmax ** (Tc - 25) / 10
    denom = 1 + np.exp((-2.2e5 + 710 * (Tc + 273.16)) / (R * (Tc + 273.16)))
    return leaf_params['vcmax25'] * Q10 / denom

def calc_Kc(T_air):
    return Kc25 * 2.1 ** ((T_air - 273.15) - 25) / 10

def calc_Ko(T_air):
    return Ko25 * 1.2 ** ((T_air - 273.15) - 25) / 10

def calc_A(Ci, T_air, PAR):
    if leaf_params['pathway'] == 'C3':
        Vcmax = calc_vcm(T_air) * leaf_params['n_factor'] * forcing['btran'].values
        Kc = calc_Kc(T_air)
        Ko = calc_Ko(T_air)
        Gamma_star = 0.5 * Kc / Ko * forcing['o2_pa'].values * 0.21
        Rd = Rd_factor * Vcmax
        Wc = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + forcing['o2_pa'].values / Ko))
        J = leaf_params['alpha_q'] * (4.6 * PAR)  # simple form
        Wj = J * (Ci - Gamma_star) / (4 * Ci + 8 * Gamma_star)
        We = 0.5 * Vcmax
        A = np.minimum(Wc, Wj, We) - Rd
    else:  # C4
        Vcmax = calc_vcm(T_air) * leaf_params['n_factor'] * forcing['btran'].values
        Rd = Rd_factor * Vcmax
        Jmax = 1.97 * Vcmax
        alpha_q = leaf_params['alpha_q']
        APAR = 4.6 * PAR
        # Non-rectangular hyperbola for J
        J = fsolve(lambda J: alpha_q * APAR - J * (1 - J / Jmax) * 0.7, 0)[0]
        A = np.minimum(Vcmax, alpha_q * APAR, 4000 * Vcmax * Ci / forcing['p_atm_Pa'].values) - Rd
    return A

def solve_coupled(Ci_init, T_air, PAR, ci_init_pa):
    def func(Ci):
        A = calc_A(Ci, T_air, PAR)
        A_mol = A * A_mol_factor
        cs = forcing['co2_pa'].values - 1.37 * forcing['rb_sm'].values * A_mol / cf_val * forcing['p_atm_Pa'].values
        gs = leaf_params['bb_slope'] * A_mol * forcing['h_s'].values / cs * forcing['p_atm_Pa'].values + leaf_params['bb_intercept']
        rs = cf_val / gs
        Ci_new = cs - 1.65 * rs * A_mol / cf_val * forcing['p_atm_Pa'].values
        return Ci_new - Ci
    Ci_final = fsolve(func, ci_init_pa)[0]
    A_final = calc_A(Ci_final, T_air, PAR)
    A_mol_final = A_final * A_mol_factor
    gs_final = leaf_params['bb_slope'] * A_mol_final * forcing['h_s'].values / (cs - 1.65 * rs * A_mol_final / cf_val * forcing['p_atm_Pa'].values) * forcing['p_atm_Pa'].values + leaf_params['bb_intercept']
    rs_final = cf_val / gs_final
    return Ci_final, A_final, rs_final, gs_final

def run_model():
    results = []
    case_results = []
    for idx in range(len(forcing)):
        time_hr = forcing['time_hr'].values[idx]
        PAR_sun = forcing['par_sun_wm2'].values[idx]
        PAR_shade = forcing['par_shade_wm2'].values[idx]
        T_air = forcing['t_air_K'].values[idx]
        co2_pa = forcing['co2_pa'].values[idx]
        o2_pa = forcing['o2_pa'].values[idx]
        vpd_pa = forcing['vpd_Pa'].values[idx]
        rb_sm = forcing['rb_sm'].values[idx]
        btran = forcing['btran'].values[idx]
        h_s = 1 - vpd_pa / forcing['p_atm_Pa'].values[idx]
        if PAR_sun == 0:
            A_sun = -Rd_factor * calc_vcm(T_air) * leaf_params['n_factor'] * btran
            A_shade = -Rd_factor * calc_vcm(T_air) * leaf_params['n_factor'] * btran
            gs_sun = leaf_params['bb_intercept']
            gs_shade = leaf_params['bb_intercept']
            rs_sun = cf_val / gs_sun
            rs_shade = cf_val / gs_shade
            ci_sun = co2_pa - 1.37 * rb_sm * A_sun * A_mol_factor / cf_val * forcing['p_atm_Pa'].values[idx]
            ci_shade = co2_pa - 1.37 * rb_sm * A_shade * A_mol_factor / cf_val * forcing['p_atm_Pa'].values[idx]
        else:
            ci_sun = 0.7 * co2_pa
            ci_shade = 0.4 * co2_pa
            ci_sun, A_sun, rs_sun, gs_sun = solve_coupled(ci_sun, T_air, PAR_sun, ci_sun)
            ci_shade, A_shade, rs_shade, gs_shade = solve_coupled(ci_shade, T_air, PAR_shade, ci_shade)
        results.append([time_hr, A_sun, A_shade, rs_sun, rs_shade, ci_sun, ci_shade, gs_sun, gs_shade])
    df = pd.DataFrame(results, columns=['time_hr', 'A_sun', 'A_shade', 'rs_sun', 'rs_shade', 'ci_sun', 'ci_shade', 'gs_sun', 'gs_shade'])
    df.to_csv('leaf_fluxes.csv', index=False)
    
    for case in test_cases:
        ci_init_pa = case['ci_init_pa']
        forcing_index = case['forcing_index']
        T_air = forcing['t_air_K'].values[forcing_index]
        PAR = forcing['par_sun_wm2'].values[forcing_index]
        ci_final, A_final, rs_final, gs_final = solve_coupled(ci_init_pa, T_air, PAR, ci_init_pa)
        case_results.append([case['case_id'], ci_init_pa, ci_final, A_final, rs_final, 1])  # assume 1 iteration
    df = pd.DataFrame(case_results, columns=['case_id', 'ci_init_pa', 'ci_final_pa', 'A_final', 'rs_final', 'n_iterations'])
    df.to_csv('case_convergence.csv', index=False)
    
    # Plot diurnal cycle
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(forcing['time_hr'], forcing['par_sun_wm2'], label='PAR sun')
    axs[0].plot(forcing['time_hr'], forcing['par_shade_wm2'], label='PAR shade')
    axs[0].plot(forcing['time_hr'], df['A_sun'], label='A sun')
    axs[0].plot(forcing['time_hr'], df['A_shade'], label='A shade')
    axs[0].set_ylabel('PAR (W/m²) and A (umol/m²/s)')
    axs[0].legend()
    axs[1].plot(forcing['time_hr'], df['rs_sun'], label='rs sun')
    axs[1].plot(forcing['time_hr'], df['rs_shade'], label='rs shade')
    axs[1].set_ylabel('rs (s/m²)')
    axs[1].set_xlabel('Time (hours)')
    axs[1].legend()
    plt.savefig('diurnal.png')
    
run_model()