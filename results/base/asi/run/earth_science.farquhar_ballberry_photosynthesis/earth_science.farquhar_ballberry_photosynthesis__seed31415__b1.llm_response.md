```python
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Constants
R = 8.314  # J/mol/K
Q10_vcmax = 2.4
Kc25 = 30  # Pa
Ko25 = 30000  # Pa
Gamma_star_factor = 0.5 * Kc25 / Ko25 * 0.21  # 0.21 is CO2/O2 ratio
Rd_factor = 0.015  # CLM-style Rd factor
D_H2O_over_CO2 = 1.37  # Diffusivity ratio
D_H2O_over_CO2_for_Ci = 1.65  # Combined diffusivity ratio for Ci calculation
cf_factor = 1e-6  # Conversion factor for molar to mass (1e-6 mol/mol)

# Load data
leaf_params_path = os.path.join('data', 'leaf_params.json')
forcing_path = os.path.join('data', 'forcing.csv')
test_cases_path = os.path.join('data', 'test_cases.json')

with open(leaf_params_path, 'r') as f:
    leaf_params = json.load(f)

forcing_df = pd.read_csv(forcing_path)
test_cases = json.load(open(test_cases_path, 'r'))

# Parse leaf parameters
pathway = leaf_params['pathway']
vcmax25 = leaf_params['vcmax25']
jmax25 = leaf_params['jmax25'] if pathway == 'C3' else leaf_params['alpha_q']
bb_slope = leaf_params['bb_slope']
bb_intercept = leaf_params['bb_intercept']
n_factor = leaf_params['n_factor']

# Function to compute Vcmax(T)
def compute_vcmax(Tc, Tc25=25):
    Q10_term = Q10_vcmax ** ((Tc - Tc25) / 10)
    denominator = 1 + np.exp((-2.2e5 + 710 * (Tc + 273.16)) / (R * (Tc + 273.16)))
    return vcmax25 * Q10_term / denominator

# Function to compute Kc(T)
def compute_Kc(Tc, Tc25=25):
    return Kc25 * 2.1 ** ((Tc - Tc25) / 10)

# Function to compute Ko(T)
def compute_Ko(Tc, Tc25=25):
    return Ko25 * 1.2 ** ((Tc - Tc25) / 10)

# Function to compute Gamma_star
def compute_Gamma_star(O2_pa, Tc, Kc, Ko):
    return Gamma_star_factor * Kc / Ko * O2_pa

# Function to compute Rd
def compute_Rd(Vcmax, Tc):
    return Rd_factor * Vcmax(Tc)

# Function to compute APAR from PAR
def compute_apar(par_wm2):
    return 4.6 * par_wm2

# Function to compute J for C3
def compute_j_c3(apar, Vcmax, Jmax, alpha_q, theta_J=0.7):
    if apar == 0:
        return 0
    J = fsolve(lambda J: theta_J * J**2 - (alpha_q * apar + J) * J + alpha_q * apar * Jmax, 0)[0]
    return np.clip(J, 0, Jmax)

# Function to compute A for C3
def compute_A_c3(Vcmax, J, Kc, Ko, O2_pa, Tc, Gamma_star, Rd):
    Wc = Vcmax * (Ci - Gamma_star) / (Ci + Kc * (1 + O2_pa / Ko))
    Wj = J * (Ci - Gamma_star) / (4 * Ci + 8 * Gamma_star)
    We = 0.5 * Vcmax
    return np.minimum(np.minimum(Wc, Wj), We) - Rd

# Function to compute A for C4
def compute_A_c4(Vcmax, alpha_q, apar, Rd):
    Wc = Vcmax
    Wj = alpha_q * apar
    We = 4000 * Vcmax * Ci / P_atm
    return np.minimum(np.minimum(Wc, Wj), We) - Rd

# Function to compute stomatal conductance
def compute_gs(A_mol, h_s, cs, p_atm, bb_slope, bb_intercept):
    return bb_slope * A_mol * h_s / cs * p_atm + bb_intercept

# Function to compute Ci from cs and gs
def compute_ci(cs, gs, p_atm, T_air, R, cf):
    rs = 1 / (gs / cf)
    return cs - D_H2O_over_CO2_for_Ci * rs * A_mol / cf * p_atm

# Function to solve the coupled system
def solve_coupled_system(ci_initial, Tc, par_sun, par_shade, p_atm, O2_pa, Vcmax, Rd, Kc, Ko, Gamma_star, h_s, rb, btran, bb_slope, bb_intercept, test_case=False):
    if test_case:
        ci = ci_initial
    else:
        ci = 0.7 * p_atm if pathway == 'C3' else 0.4 * p_atm
    A_sun = np.zeros_like(par_sun)
    A_shade = np.zeros_like(par_shade)
    Ci_sun = np.zeros_like(par_sun)
    Ci_shade = np.zeros_like(par_shade)
    gs_sun = np.zeros_like(par_sun)
    gs_shade = np.zeros_like(par_shade)
    rs_sun = np.zeros_like(par_sun)
    rs_shade = np.zeros_like(par_shade)
    n_iterations = 0

    for i in range(len(par_sun)):
        par_sun_i = par_sun[i]
        par_shade_i = par_shade[i]
        Tc_i = Tc[i]
        p_atm_i = p_atm[i]
        O2_pa_i = O2_pa[i]
        rb_i = rb[i]
        btran_i = btran[i]
        h_s_i = h_s[i]

        if par_sun_i == 0:
            A_sun[i] = -Rd(Tc_i)
            gs_sun[i] = bb_intercept
            rs_sun[i] = 1 / (gs_sun[i] / cf)
        else:
            Vcmax_i = compute_vcmax(Tc_i) * n_factor * btran_i
            Kc_i = compute_Kc(Tc_i)
            Ko_i = compute_Ko(Tc_i)
            Gamma_star_i = compute_Gamma_star(O2_pa_i, Tc_i, Kc_i, Ko_i)
            Rd_i = compute_Rd(Vcmax_i, Tc_i)

            if pathway == 'C3':
                apar_sun_i = compute_apar(par_sun_i)
                J_sun = compute_j_c3(apar_sun_i, Vcmax_i, jmax25, alpha_q, theta_J=0.7)
                A_sun[i] = compute_A_c3(Vcmax_i, J_sun, Kc_i, Ko_i, O2_pa_i, Tc_i, Gamma_star_i, Rd_i)
            else:
                apar_sun_i = compute_apar(par_sun_i)
                A_sun[i] = compute_A_c4(Vcmax_i, alpha_q, apar_sun_i, Rd_i)

            A_sun_mol = A_sun[i] * 1e-6
            cf_i = p_atm_i / (R * Tc_i)
            cs_sun = p_atm_i - 1.37 * rb_i * A_sun_mol / cf_i * p_atm_i
            gs_sun[i] = compute_gs(A_sun_mol, h_s_i, cs_sun, p_atm_i, bb_slope, bb_intercept)
            rs_sun[i] = 1 / (gs_sun[i] / cf_i)
            Ci_sun[i] = cs_sun - D_H2O_over_CO2_for_Ci * rs_sun[i] * A_sun_mol / cf_i * p_atm_i

        if par_shade_i == 0:
            A_shade[i] = -Rd(Tc_i)
            gs_shade[i] = bb_intercept
            rs_shade[i] = 1 / (gs_shade[i] / cf)
        else:
            Vcmax_i = compute_vcmax(Tc_i) * n_factor * btran_i
            Kc_i = compute_Kc(Tc_i)
            Ko_i = compute_Ko(Tc_i)
            Gamma_star_i = compute_Gamma_star(O2_pa_i, Tc_i, Kc_i, Ko_i)
            Rd_i = compute_Rd(Vcmax_i, Tc_i)

            if pathway == 'C3':
                apar_shade_i = compute_apar(par_shade_i)
                J_shade = compute_j_c3(apar_shade_i, Vcmax_i, jmax25, alpha_q, theta_J=0.7)
                A_shade[i] = compute_A_c3(Vcmax_i, J_shade, Kc_i, Ko_i, O2_pa_i, Tc_i, Gamma_star_i, Rd_i)
            else:
                apar_shade_i = compute_apar(par_shade_i)
                A_shade[i] = compute_A_c4(Vcmax_i, alpha_q, apar_shade_i, Rd_i)

            A_shade_mol = A_shade[i] * 1e-6
            cf_i = p_atm_i / (R * Tc_i)
            cs_shade = p_atm_i - 1.37 * rb_i * A_shade_mol / cf_i * p_atm_i
            gs_shade[i] = compute_gs(A_shade_mol, h_s_i, cs_shade, p_atm_i, bb_slope, bb_intercept)
            rs_shade[i] = 1 / (gs_shade[i] / cf_i)
            Ci_shade[i] = cs_shade - D_H2O_over_CO2_for_Ci * rs_shade[i] * A_shade_mol / cf_i * p_atm_i

        if not test_case:
            ci = Ci_sun[i]
            n_iterations = 0
            while np.abs(ci - Ci_sun[i]) / ci > 1e-3:
                ci = Ci_sun[i]
                n_iterations += 1
                if n_iterations > 100:
                    break
            if n_iterations > 100:
                ci = Ci_sun[i]

    return A_sun, A_shade, Ci_sun, Ci_shade, gs_sun, gs_shade, rs_sun, rs_shade, n_iterations

# Main processing
forcing_df['time_hr'] = pd.to_datetime(forcing_df['time_hr']).dt.hour
forcing_df['T_air_K'] = forcing_df['T_air_K'].astype(float)
forcing_df['p_atm_Pa'] = forcing_df['p_atm_Pa'].astype(float)
forcing_df['co2_pa'] = forcing_df['co2_pa'].astype(float)
forcing_df['o2_pa'] = forcing_df['o2_pa'].astype(float)
forcing_df['vpd_Pa'] = forcing_df['vpd_Pa'].astype(float)
forcing_df['rb_sm'] = forcing_df['rb_sm'].astype(float)
forcing_df['btran'] = forcing_df['btran'].astype(float)
forcing_df['h_s'] = 1 - (forcing_df['vpd_Pa'] / forcing_df['p_atm_Pa'])

results = []
for index, row in forcing_df.iterrows():
    Tc = row['T_air_K']
    par_sun = row['par_sun_wm2']
    par_shade = row['par_shade_wm2']
    p_atm = row['p_atm_Pa']
    O2_pa = row['o2_pa']
    rb = row['rb_sm']
    btran = row['btran']
    h_s = row['h_s']

    A_sun, A_shade, Ci_sun, Ci_shade, gs_sun, gs_shade, rs_sun, rs_shade, n_iterations = solve_coupled_system(
        0.7 * p_atm if pathway == 'C3' else 0.4 * p_atm, Tc, par_sun, par_shade, p_atm, O2_pa, compute_vcmax, compute_Rd, compute_Kc, compute_Ko, compute_Gamma_star, h_s, rb, btran, bb_slope, bb_intercept, test_case=False
    )

    results.append({
        'time_hr': row['time_hr'],
        'A_sun': A_sun,
        'A_shade': A_shade,
        'rs_sun': rs_sun,
        'rs_shade': rs_shade,
        'ci_sun': Ci_sun,
        'ci_shade': Ci_shade,
        'gs_sun': gs_sun,
        'gs_shade': gs_shade
    })

# Save leaf fluxes
leaf_fluxes_df = pd.DataFrame(results)
leaf_fluxes_df.to_csv('results/leaf_fluxes.csv', index=False)

# Process test cases
case_results = []
for case in test_cases:
    case_id = case['case_id']
    ci_init_pa = case['ci_init_pa']
    forcing_index = case['forcing_index']
    row = forcing_df.iloc[forcing_index]
    Tc = row['T_air_K']
    par_sun = row['par_sun_wm2']
    par_shade = row['par_shade_wm2']
    p_atm = row['p_atm_Pa']
    O2_pa = row['o2_pa']
    rb = row['rb_sm']
    btran = row['btran']
    h_s = row['h_s']

    A_sun, A_shade, Ci_sun, Ci_shade, gs_sun, gs_shade, rs_sun, rs_shade, n_iterations = solve_coupled_system(
        ci_init_pa, Tc, par_sun, par_shade, p_atm, O2_pa, compute_vcmax, compute_Rd, compute_Kc, compute_Ko, compute_Gamma_star, h_s, rb, btran, bb_slope, bb_intercept, test_case=True
    )

    case_results.append({
        'case_id': case_id,
        'ci_init_pa': ci_init_pa,
        'ci_final_pa': Ci_sun,
        'A_final': A_sun,
        'rs_final': rs_sun,
        'n_iterations': n_iterations
    })

case_results_df = pd.DataFrame(case_results)
case_results_df.to_csv('results/case_convergence.csv', index=False)

# Plot diurnal cycle
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(forcing_df['time_hr'], forcing_df['A_sun'], label='A_sun')
plt.plot(forcing_df['time_hr'], forcing_df['A_shade'], label='A_shade')
plt.xlabel('Time (hr)')
plt.ylabel('A (umol CO2 m^-2 s^-1)')
plt.title('Diurnal A_sun and A_shade')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(forcing_df['time_hr'], forcing_df['rs_sun'], label='rs_sun')
plt.plot(forcing_df['time_hr'], forcing_df['rs_shade'], label='rs_shade')
plt.xlabel('Time (hr)')
plt.ylabel('rs (s m^-1)')
plt.title('Diurnal rs_sun and rs_shade')
plt.legend()

plt.tight_layout()
plt.savefig('results/diurnal.png')
plt.close()
```