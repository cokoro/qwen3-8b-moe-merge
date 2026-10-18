```python
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

def analyze():
    # Load data
    surface_samples = pd.read_csv('data/surface_samples.csv')
    curve_samples = pd.read_csv('data/curve_samples.csv')
    tube_scan = pd.read_csv('data/tube_scan.csv')
    kinetic_metric = pd.read_csv('data/kinetic_metric.csv')
    temperature_grid = pd.read_csv('data/temperature_grid.csv')
    isotope_cases = pd.read_csv('data/isotope_cases.csv')
    manifest = json.load(open('data/system_manifest.json'))
    
    # Step 1: Reconstruct potential along the corridor
    def interpolate_scan(group):
        q = group['q_offset'].values
        v = group['potential_kj_mol'].values
        coeffs = np.polyfit(q, v, 2)
        return coeffs
    
    corridor_potential = tube_scan.groupby('point_id').apply(interpolate_scan)
    corridor_potential = corridor_potential.reset_index()
    corridor_potential.columns = ['point_id', 'q2_coeff', 'q_coeff', 'const_coeff']
    
    # Step 2: Determine wells and saddle
    def find_wells_and_saddle(potential):
        # Find reactant well
        reactant_index = potential['s_coord'].idxmin()
        reactant_s = potential.loc[reactant_index, 's_coord']
        reactant_v = potential.loc[reactant_index, 'potential_kj_mol']
        
        # Find product well
        product_index = potential['s_coord'].idxmax()
        product_s = potential.loc[product_index, 's_coord']
        product_v = potential.loc[product_index, 'potential_kj_mol']
        
        # Find saddle point
        saddle_index = potential['potential_kj_mol'].idxmax()
        saddle_s = potential.loc[saddle_index, 's_coord']
        saddle_v = potential.loc[saddle_index, 'potential_kj_mol']
        
        return reactant_s, reactant_v, product_s, product_v, saddle_s, saddle_v
    
    reactant_s, reactant_v, product_s, product_v, saddle_s, saddle_v = find_wells_and_saddle(corridor_potential)
    
    # Step 3: Compute harmonic oscillator free energy
    def harmonic_free_energy(omega, T):
        hc_kJ_mol_per_cm = manifest['hc_kJ_mol_per_cm']
        R_kJ_mol_K = manifest['R_kJ_mol_K']
        return 0.5 * hc_kJ_mol_per_cm * omega + R_kJ_mol_K * T * np.log(1 - np.exp(-hc_kJ_mol_per_cm * omega / (R_kJ_mol_K * T)))
    
    # Step 4: Conventional TST and VTST
    def compute_rates(temperature):
        # Compute harmonic oscillator frequencies
        omega_reactant_long = 0.0
        omega_reactant_trans = 0.0
        omega_saddle = 0.0
        
        # Compute canonical rate
        k_canonical = 0.0
        
        # Compute variational rate
        k_variational = 0.0
        
        return k_canonical, k_variational
    
    # Step 5: Compute tunneling factor
    def compute_tunneling_factor(temperature):
        # Compute tunneling factor
        tunneling_factor = 0.0
        return tunneling_factor
    
    # Step 6: Compute action profiles
    def compute_action_profiles(temperature):
        # Compute action profiles
        action_mep = 0.0
        action_sct = 0.0
        return action_mep, action_sct
    
    # Step 7: Compute isotope effects
    def compute_isotope_effects(isotope_case):
        # Compute isotope effects
        log10_k_vtst_sct_s_inv = 0.0
        return log10_k_vtst_sct_s_inv
    
    # Write outputs
    free_energy_profile = pd.DataFrame(columns=['T_K', 'point_id', 's_coord', 'x', 'y', 'V_mep_kj_mol', 'omega_perp_cm1', 'free_energy_rel_kj_mol', 'is_variational_max', 'is_conventional_saddle'])
    rate_constants = pd.DataFrame(columns=['T_K', 'log10_k_tst_s_inv', 'log10_k_vtst_s_inv', 'log10_k_vtst_sct_s_inv', 'kappa_mep', 'kappa_sct', 's_vtst', 'barrier_tst_kj_mol', 'barrier_vtst_kj_mol', 'crossover_T_K'])
    transmission_profile = pd.DataFrame(columns=['energy_id', 'energy_kj_mol', 'action_mep', 'action_sct', 'transmission_mep', 'transmission_sct', 'action_ratio_sct_over_mep'])
    path_samples = pd.DataFrame(columns=['energy_id', 'path_step', 'energy_kj_mol', 'x', 'y', 'potential_kj_mol'])
    isotope_effects = pd.DataFrame(columns=['case_id', 'isotopologue', 'longitudinal_mass_scale', 'transverse_mass_scale', 'T_K', 'log10_k_tst_s_inv', 'log10_k_vtst_s_inv', 'log10_k_vtst_sct_s_inv', 'kappa_sct', 's_vtst', 'barrier_vtst_kj_mol', 'crossover_T_K', 'log10_KIE_sct_vs_light'])
    isotope_transmission_profile = pd.DataFrame(columns=['case_id', 'isotopologue', 'longitudinal_mass_scale', 'transverse_mass_scale', 'energy_id', 'energy_kj_mol', 'action_mep', 'action_sct', 'transmission_mep', 'transmission_sct', 'action_ratio_sct_over_mep'])
    diagnostics = {}
    
    # Write results
    free_energy_profile.to_csv('results/free_energy_profile.csv', index=False)
    rate_constants.to_csv('results/rate_constants.csv', index=False)
    transmission_profile.to_csv('results/transmission_profile.csv', index=False)
    path_samples.to_csv('results/path_samples.csv', index=False)
    isotope_effects.to_csv('results/isotope_effects.csv', index=False)
    isotope_transmission_profile.to_csv('results/isotope_transmission_profile.csv', index=False)
    with open('results/diagnostics.json', 'w') as f:
        json.dump(diagnostics, f)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(temperature_grid['T_K'], rate_constants['log10_k_tst_s_inv'], label='TST')
    plt.plot(temperature_grid['T_K'], rate_constants['log10_k_vtst_s_inv'], label='VTST')
    plt.plot(temperature_grid['T_K'], rate_constants['log10_k_vtst_sct_s_inv'], label='VTST+Tunneling')
    plt.xlabel('Temperature (K)')
    plt.ylabel('log10(k/s)')
    plt.legend()
    plt.savefig('results/rate_plot.png')
    plt.close()
```