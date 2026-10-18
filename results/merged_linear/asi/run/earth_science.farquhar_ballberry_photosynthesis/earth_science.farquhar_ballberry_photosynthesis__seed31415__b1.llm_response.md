```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def read_json(path):
    with open(path) as f:
        return json.load(f)

def write_csv(df, path):
    df.to_csv(path, index=False)

def plot_results(df, path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df['time_hr'], df['A_sun'], label='Sun')
    plt.plot(df['time_hr'], df['A_shade'], label='Shade')
    plt.xlabel('Time (hr)')
    plt.ylabel('A (umol CO2 m^-2 s^-1)')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df['time_hr'], df['rs_sun'], label='Sun')
    plt.plot(df['time_hr'], df['rs_shade'], label='Shade')
    plt.xlabel('Time (hr)')
    plt.ylabel('rs (s m^-1)')
    plt.legend()
    plt.savefig(path)
    plt.close()

def compute_leaf_fluxes(leaf_params, forcing, test_cases):
    # Read parameters
    pathway = leaf_params['pathway']
    vcmax25 = leaf_params['vcmax25']
    jmax25 = leaf_params['jmax25'] if pathway == 'C3' else leaf_params['alpha_q']
    bb_slope = leaf_params['bb_slope']
    bb_intercept = leaf_params['bb_intercept']
    n_factor = leaf_params['n_factor']

    # Read forcing data
    df = pd.DataFrame(forcing)
    time_hr = df['time_hr'].values
    par_sun = df['par_sun_wm2'].values
    par_shade = df['par_shade_wm2'].values
    t_air_K = df['t_air_K'].values
    p_atm_Pa = df['p_atm_Pa'].values
    co2_pa = df['co2_pa'].values
    o2_pa = df['o2_pa'].values
    vpd_Pa = df['vpd_Pa'].values
    rb_sm = df['rb_sm'].values
    btran = df['btran'].values

    # Initialize output arrays
    A_sun = np.zeros_like(par_sun)
    A_shade = np.zeros_like(par_shade)
    rs_sun = np.zeros_like(par_sun)
    rs_shade = np.zeros_like(par_shade)
    ci_sun = np.zeros_like(par_sun)
    ci_shade = np.zeros_like(par_shade)
    gs_sun = np.zeros_like(par_sun)
    gs_shade = np.zeros_like(par_shade)

    # Compute constants
    R = 8.314  # J/mol/K
    cf = p_atm_Pa / (R * t_air_K)  # mol/m^3
    h_s = (p_atm_Pa - vpd_Pa) / p_atm_Pa  # relative humidity
    molar_mass = 44.01  # g/mol CO2
    R_d = 0.015  # mol CO2/m^2/s at 25°C

    # Compute temperature-dependent parameters
    Tc = t_air_K
    Q10_vcmax = 2.4
    Kc25 = 30.0
    Ko25 = 30000.0

    def vcmax(T):
        return (vcmax25 * Q10_vcmax ** ((T - 25) / 10) /
                (1 + np.exp((-2.2e5 + 710 * (T + 273.16)) / (R * (T + 273.16)))))

    def Kc(T):
        return Kc25 * 2.1 ** ((T - 25) / 10)

    def Ko(T):
        return Ko25 * 1.2 ** ((T - 25) / 10)

    def Gamma_star(O2_pa):
        return 0.5 * Kc(Tc) / Ko(Tc) * O2_pa * 0.21

    def Rd(Vcmax):
        return R_d * Vcmax

    # Compute for each time step
    for i in range(len(time_hr)):
        # Sun side
        if par_sun[i] > 0:
            A_sun[i] = 0
            Ci_sun = co2_pa[i] * 0.7 if pathway == 'C3' else co2_pa[i] * 0.4
            for _ in range(100):
                # Compute Farquhar photosynthesis
                Vcmax_eff = vcmax(Tc) * n_factor * btran[i]
                if pathway == 'C3':
                    Jmax = 1.97 * Vcmax_eff
                    APAR = 4.6 * par_sun[i]
                    if APAR < 0:
                        J = 0
                    else:
                        # Non-rectangular hyperbola
                        theta_J = 0.7
                        J = np.roots([theta_J, - (jmax25 * APAR + Jmax), jmax25 * APAR * Jmax])[0]
                    Wc = Vcmax_eff * (Ci_sun - Gamma_star(o2_pa[i])) / (Ci_sun + Kc(Tc) * (1 + o2_pa[i] / Ko(Tc)))
                    Wj = J * (Ci_sun - Gamma_star(o2_pa[i])) / (4 * Ci_sun + 8 * Gamma_star(o2_pa[i]))
                    We = 0.5 * Vcmax_eff
                    A = min(Wc, Wj, We) - Rd(Vcmax_eff)
                else:  # C4
                    J = alpha_q * par_sun[i]
                    Wc = Vcmax_eff
                    Wj = alpha_q * par_sun[i]
                    We = 4000 * Vcmax_eff * Ci_sun / p_atm_Pa[i]
                    A = min(Wc, Wj, We) - Rd(Vcmax_eff)
                A_sun[i] = A

                # Compute Ball-Berry conductance
                A_mol = A_sun[i] * 1e-6
                cs = co2_pa[i] - 1.37 * rb_sm[i] * A_mol / cf[i] * p_atm_Pa[i]
                gs_sun[i] = bb_slope * A_mol * h_s[i] / cs * p_atm_Pa[i] + bb_intercept
                rs_sun[i] = cf[i] / gs_sun[i]
                Ci_sun = cs - 1.65 * rs_sun[i] * A_mol / cf[i] * p_atm_Pa[i]
        else:
            A_sun[i] = -Rd(vcmax(Tc) * n_factor * btran[i])
            gs_sun[i] = bb_intercept
            rs_sun[i] = cf[i] / gs_sun[i]
            ci_sun[i] = cs - 1.65 * rs_sun[i] * A_mol / cf[i] * p_atm_Pa[i]

        # Shade side
        if par_shade[i] > 0:
            A_shade[i] = 0
            Ci_shade = co2_pa[i] * 0.7 if pathway == 'C3' else co2_pa[i] * 0.4
            for _ in range(100):
                # Compute Farquhar photosynthesis
                Vcmax_eff = vcmax(Tc) * n_factor * btran[i]
                if pathway == 'C3':
                    Jmax = 1.97 * Vcmax_eff
                    APAR = 4.6 * par_shade[i]
                    if APAR < 0:
                        J = 0
                    else:
                        # Non-rectangular hyperbola
                        theta_J = 0.7
                        J = np.roots([theta_J, - (jmax25 * APAR + Jmax), jmax25 * APAR * Jmax])[0]
                    Wc = Vcmax_eff * (Ci_shade - Gamma_star(o2_pa[i])) / (Ci_shade + Kc(Tc) * (1 + o2_pa[i] / Ko(Tc)))
                    Wj = J * (Ci_shade - Gamma_star(o2_pa[i])) / (4 * Ci_shade + 8 * Gamma_star(o2_pa[i]))
                    We = 0.5 * Vcmax_eff
                    A = min(Wc, Wj, We) - Rd(Vcmax_eff)
                else:  # C4
                    J = alpha_q * par_shade[i]
                    Wc = Vcmax_eff
                    Wj = alpha_q * par_shade[i]
                    We = 4000 * Vcmax_eff * Ci_shade / p_atm_Pa[i]
                    A = min(Wc, Wj, We) - Rd(Vcmax_eff)
                A_shade[i] = A

                # Compute Ball-Berry conductance
                A_mol = A_shade[i] * 1e-6
                cs = co2_pa[i] - 1.37 * rb_sm[i] * A_mol / cf[i] * p_atm_Pa[i]
                gs_shade[i] = bb_slope * A_mol * h_s[i] / cs * p_atm_Pa[i] + bb_intercept
                rs_shade[i] = cf[i] / gs_shade[i]
                Ci_shade = cs - 1.65 * rs_shade[i] * A_mol / cf[i] * p_atm_Pa[i]
        else:
            A_shade[i] = -Rd(vcmax(Tc) * n_factor * btran[i])
            gs_shade[i] = bb_intercept
            rs_shade[i] = cf[i] / gs_shade[i]
            ci_shade[i] = cs - 1.65 * rs_shade[i] * A_mol / cf[i] * p_atm_Pa[i]

    # Create output dataframe
    df_out = pd.DataFrame({
        'time_hr': time_hr,
        'A_sun': A_sun,
        'A_shade': A_shade,
        'rs_sun': rs_sun,
        'rs_shade': rs_shade,
        'ci_sun': ci_sun,
        'ci_shade': ci_shade,
        'gs_sun': gs_sun,
        'gs_shade': gs_shade
    })
    write_csv(df_out, 'results/leaf_fluxes.csv')

    # Process test cases
    df_test = pd.DataFrame(test_cases)
    case_id = df_test['case_id'].values
    ci_init_pa = df_test['ci_init_pa'].values
    forcing_index = df_test['forcing_index'].values

    case_results = []
    for i in range(len(case_id)):
        ci_init = ci_init_pa[i]
        forcing_idx = forcing_index[i]
        ci_final = ci_init
        A_final = 0
        rs_final = 0
        n_iterations = 0
        for _ in range(100):
            # Compute Farquhar photosynthesis
            Vcmax_eff = vcmax(Tc) * n_factor * btran[forcing_idx]
            if pathway == 'C3':
                Jmax = 1.97 * Vcmax_eff
                APAR = 4.6 * par_sun[forcing_idx]
                if APAR < 0:
                    J = 0
                else:
                    # Non-rectangular hyperbola
                    theta_J = 0.7
                    J = np.roots([theta_J, - (jmax25 * APAR + Jmax), jmax25 * APAR * Jmax])[0]
                Wc = Vcmax_eff * (ci_init - Gamma_star(o2_pa[forcing_idx])) / (ci_init + Kc(Tc) * (1 + o2_pa[forcing_idx] / Ko(Tc)))
                Wj = J * (ci_init - Gamma_star(o2_pa[forcing_idx])) / (4 * ci_init + 8 * Gamma_star(o2_pa[forcing_idx]))
                We = 0.5 * Vcmax_eff
                A = min(Wc, Wj, We) - Rd(Vcmax_eff)
            else:  # C4
                J = alpha_q * par_sun[forcing_idx]
                Wc = Vcmax_eff
                Wj = alpha_q * par_sun[forcing_idx]
                We = 4000 * Vcmax_eff * ci_init / p_atm_Pa[forcing_idx]
                A = min(Wc, Wj, We) - Rd(Vcmax_eff)
            A_final = A

            # Compute Ball-Berry conductance
            A_mol = A_final * 1e-6
            cs = co2_pa[forcing_idx] - 1.37 * rb_sm[forcing_idx] * A_mol / cf[forcing_idx] * p_atm_Pa[forcing_idx]
            gs = bb_slope * A_mol * h_s[forcing_idx] / cs * p_atm_Pa[forcing_idx] + bb_intercept
            rs = cf[forcing_idx] / gs
            ci_final = cs - 1.65 * rs * A_mol / cf[forcing_idx] * p_atm_Pa[forcing_idx]
            n_iterations += 1
            if np.abs(ci_final - ci_init) / ci_init < 1e-3:
                break
            ci_init = ci_final

        case_results.append({
            'case_id': case_id[i],
            'ci_init_pa': ci_init_pa[i],
            'ci_final_pa': ci_final,
            'A_final': A_final,
            'rs_final': rs,
            'n_iterations': n_iterations
        })

    df_case = pd.DataFrame(case_results)
    write_csv(df_case, 'results/case_convergence.csv')

    plot_results(df_out, 'results/diurnal.png')

if __name__ == '__main__':
    leaf_params = read_json('data/leaf_params.json')
    forcing = read_json('data/forcing.csv')
    test_cases = read_json('data/test_cases.json')
    compute_leaf_fluxes(leaf_params, forcing, test_cases)
```