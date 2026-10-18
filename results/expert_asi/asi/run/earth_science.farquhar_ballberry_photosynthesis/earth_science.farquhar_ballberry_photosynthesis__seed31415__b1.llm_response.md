```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_csv(path):
    return pd.read_csv(path)

def save_csv(path, df):
    df.to_csv(path, index=False)

def compute_v_cmax(T, Vcmax25, Q10_vcmax=2.4, R=8.314):
    Tc = T - 273.16
    numerator = np.exp(-2.2e5 + 710*(Tc + 273.16)) * 2.2e5
    denominator = 1 + np.exp(-2.2e5 + 710*(Tc + 273.16))
    return Vcmax25 * Q10_vcmax**((Tc - 25)/10) / (numerator / denominator)

def compute_kc(T, Kc25=30):
    return Kc25 * 2.1**((T - 25)/10)

def compute_ko(T, Ko25=30000):
    return Ko25 * 1.2**((T - 25)/10)

def compute_gamma_star(Kc, Ko, O2_pa):
    return 0.5 * Kc / Ko * O2_pa * 0.21

def compute_rd(Vcmax):
    return 0.015 * Vcmax

def compute_apar(par_wm2, par_shade_wm2):
    return 4.6 * (par_wm2 + par_shade_wm2)

def compute_j(Vcmax, alpha_q, apar, theta_J=0.7, jmax=1.97):
    # Solve quadratic equation: theta_J * J^2 - (alpha_q * apar + Jmax)*J + alpha_q * apar * Jmax = 0
    Jmax = jmax * Vcmax
    a = theta_J
    b = -(alpha_q * apar + Jmax)
    c = alpha_q * apar * Jmax
    discriminant = b**2 - 4*a*c
    J = (-b - np.sqrt(discriminant)) / (2*a)
    return np.clip(J, 0, Jmax)

def compute_farquhar_C3(Ci, T, Vcmax, Kc, Ko, O2_pa, alpha_q, Rd, n_factor, btran):
    Vcmax_eff = Vcmax * n_factor * btran
    Gamma_star = compute_gamma_star(Kc, Ko, O2_pa)
    Wc = Vcmax_eff * (Ci - Gamma_star) / (Ci + Kc * (1 + O2_pa / Ko))
    J = compute_j(Vcmax_eff, alpha_q, apar, theta_J=0.7, jmax=1.97)
    We = 0.5 * Vcmax_eff
    A = np.min([Wc, J, We]) - Rd
    return A

def compute_farquhar_C4(Ci, T, Vcmax, alpha_q, Rd, n_factor, btran, p_atm):
    Vcmax_eff = Vcmax * n_factor * btran
    J = alpha_q * apar
    We = 4000 * Vcmax_eff * Ci / p_atm
    A = np.min([Vcmax_eff, J, We]) - Rd
    return A

def compute_ball_berry(A_mol, h_s, cs, p_atm, m, b0):
    gs = m * A_mol * h_s / cs * p_atm + b0
    return gs

def compute_cs(ca, rb, A_mol, cf, p_atm):
    return ca - 1.37 * rb * A_mol / cf * p_atm

def compute_ci(cs, rs, A_mol, cf, p_atm):
    return cs - 1.65 * rs * A_mol / cf * p_atm

def solve_coupled(Ci_init, ca, T, p_atm, Vcmax25, pathway, alpha_q, m, b0, n_factor, btran, rb, h_s, apar, Rd, Kc25, Ko25, O2_pa, forcing_index):
    Tc = T - 273.16
    Vcmax = compute_v_cmax(T, Vcmax25)
    Kc = compute_kc(T, Kc25)
    Ko = compute_ko(T, Ko25)
    if pathway == 'C3':
        Rd = compute_rd(Vcmax)
        A = compute_farquhar_C3(Ci_init, T, Vcmax, Kc, Ko, O2_pa, alpha_q, Rd, n_factor, btran)
    else:
        Rd = compute_rd(Vcmax)
        A = compute_farquhar_C4(Ci_init, T, Vcmax, alpha_q, Rd, n_factor, btran, p_atm)
    A_mol = A * 1e-6
    cf = p_atm / (R * T)
    cs = compute_cs(ca, rb, A_mol, cf, p_atm)
    gs = compute_ball_berry(A_mol, h_s, cs, p_atm, m, b0)
    rs = cf / gs
    Ci = compute_ci(cs, rs, A_mol, cf, p_atm)
    return Ci, A, gs, rs, rs

def main():
    params = load_json('data/leaf_params.json')
    forcing = load_csv('data/forcing.csv')
    test_cases = load_json('data/test_cases.json')

    T = forcing['t_air_K'].values
    p_atm = forcing['p_atm_Pa'].values
    ca = forcing['co2_pa'].values
    o2_pa = forcing['o2_pa'].values
    vpd_pa = forcing['vpd_Pa'].values
    rb_sm = forcing['rb_sm'].values
    btran = forcing['btran'].values
    par_sun_wm2 = forcing['par_sun_wm2'].values
    par_shade_wm2 = forcing['par_shade_wm2'].values
    apar = compute_apar(par_sun_wm2, par_shade_wm2)
    h_s = 1 - vpd_pa / p_atm  # relative humidity

    leaf_fluxes = []
    case_convergence = []

    for idx, row in forcing.iterrows():
        if row['par_sun_wm2'] == 0:
            A_sun = -Rd
            A_shade = -Rd
            rs_sun = cf / b0
            rs_shade = cf / b0
            ci_sun = 0.7 * ca[idx]
            ci_shade = 0.7 * ca[idx]
            gs_sun = b0
            gs_shade = b0
        else:
            # Solve coupled system
            Ci_sun = solve_coupled(Ci_init=0.7 * ca[idx], ca=ca[idx], T=T[idx], p_atm=p_atm[idx], Vcmax25=params['vcmax25'], pathway=params['pathway'], alpha_q=params['alpha_q'], m=params['bb_slope'], b0=params['bb_intercept'], n_factor=params['n_factor'], btran=btran[idx], rb=rb_sm[idx], h_s=h_s[idx], apar=apar[idx], Rd=compute_rd(compute_v_cmax(T[idx], params['vcmax25'])), Kc25=params['Kc25'], Ko25=params['Ko25'], O2_pa=o2_pa[idx], forcing_index=idx)
            Ci_shade = solve_coupled(Ci_init=0.7 * ca[idx], ca=ca[idx], T=T[idx], p_atm=p_atm[idx], Vcmax25=params['vcmax25'], pathway=params['pathway'], alpha_q=params['alpha_q'], m=params['bb_slope'], b0=params['bb_intercept'], n_factor=params['n_factor'], btran=btran[idx], rb=rb_sm[idx], h_s=h_s[idx], apar=apar[idx], Rd=compute_rd(compute_v_cmax(T[idx], params['vcmax25'])), Kc25=params['Kc25'], Ko25=params['Ko25'], O2_pa=o2_pa[idx], forcing_index=idx)
            # Save results
            leaf_fluxes.append({
                'time_hr': row['time_hr'],
                'A_sun': A_sun,
                'A_shade': A_shade,
                'rs_sun': rs_sun,
                'rs_shade': rs_shade,
                'ci_sun': ci_sun,
                'ci_shade': ci_shade,
                'gs_sun': gs_sun,
                'gs_shade': gs_shade
            })
    save_csv('results/leaf_fluxes.csv', pd.DataFrame(leaf_fluxes))

    # Plot diurnal cycle
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.plot(forcing['time_hr'], forcing['par_sun_wm2'], 'r', label='PAR')
    ax1.set_ylabel('PAR (W/m²)')
    ax2.plot(forcing['time_hr'], forcing['vpd_Pa'], 'g', label='VPD')
    ax2.set_ylabel('VPD (Pa)')
    plt.show()

if __name__ == '__main__':
    main()
```