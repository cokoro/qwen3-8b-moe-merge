```bash
import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

with open('data/physics_manifest.json') as f:
    physics_manifest = json.load(f)

mu_e = 1.358

def solve_white_dwarf_branch(mu_e):
    # Define the governing equations
    def ode(y, r, mu_e):
        rho, P = y
        dPdr = -4 * np.pi * r * rho * P
        dRhoDr = -4 * np.pi * r * rho * P / (3 * P)
        return np.array([dRhoDr, dPdr])
    
    # Solve the ODE for a dense stable branch
    r0 = physics_manifest['regular_center_radius_km'] * 1e-3
    r_end = physics_manifest['surface_radius_km'] * 1e-3
    r = np.linspace(r0, r_end, 1000)
    sol = scipy.integrate.solve_ivp(ode, [r0, r_end], [1e5, 1e15], args=(mu_e,), t_eval=r)
    return sol.y[0], sol.y[1]

def scale_model(model, mu_e):
    # Scale the model using the physical scaling implied by mu_e
    M = physics_manifest['mass_scale_solar'] * mu_e**(-2)
    R = physics_manifest['radius_scale_km'] * mu_e**(-1)
    return model * R / physics_manifest['radius_scale_km'], model * M / physics_manifest['mass_scale_solar']

def compute_residuals(model, mr_observations):
    # Compute the weighted residuals over the noisy M-R observations
    w = 1 / mr_observations['uncertainty_solar'].values
    return np.sqrt(w @ (model['mass_solar'] - mr_observations['mass_solar'])**2)

def fit_mu_e():
    # Fit the hidden parameter mu_e by optimizing the residuals
    def objective(mu_e):
        model = solve_white_dwarf_branch(mu_e)
        scaled_model = scale_model(model, mu_e)
        return compute_residuals(scaled_model, mr_observations)
    result = scipy.optimize.minimize(objective, 1.358, bounds=((1.3, 2.0),))
    return result.x[0]

mu_e = fit_mu_e()
model = solve_white_dwarf_branch(mu_e)
scaled_model = scale_model(model, mu_e)

# Save the reconstructed M-R curve
mr_reconstruction = pd.DataFrame({
    'mass_solar': scaled_model[0],
    'radius_km': scaled_model[1]
})
mr_reconstruction.to_csv('results/mr_reconstruction.csv', index=False)

# Save the reconstructed profiles
density_fraction = np.zeros((len(profile_masses), len(radius_fraction_grid)))
enclosed_mass_fraction = np.zeros((len(profile_masses), len(radius_fraction_grid)))
for i, mass in enumerate(profile_masses):
    r = physics_manifest['radius_scale_km'] * mu_e**(-1)
    r = np.linspace(0, r, len(radius_fraction_grid))
    y0 = np.array([1e5, 1e15])
    sol = scipy.integrate.solve_ivp(ode, [0, r[-1]], y0, args=(mu_e,), t_eval=r)
    density_fraction[i] = sol.y[0] / sol.y[0][0]
    enclosed_mass_fraction[i] = sol.y[1] / sol.y[1][0]
np.savez('results/profiles_reconstruction.npz', density_fraction=density_fraction, enclosed_mass_fraction=enclosed_mass_fraction)

# Save the summary
summary = {
    'mass_limit_solar': scaled_model[0][-1],
    'radius_at_0p6Msun_km': scaled_model[1][np.abs(scaled_model[0] - 0.6).argmin()],
    'radius_at_1p0Msun_km': scaled_model[1][np.abs(scaled_model[0] - 1.0).argmin()],
    'radius_at_1p2Msun_km': scaled_model[1][np.abs(scaled_model[0] - 1.2).argmin()],
    'radius_at_1p3Msun_km': scaled_model[1][np.abs(scaled_model[0] - 1.3).argmin()]
}
with open('results/summary.json', 'w') as f:
    json.dump(summary, f)

# Plot the reconstruction
plt.figure(figsize=(10, 5))
plt.plot(mr_reconstruction['mass_solar'], mr_reconstruction['radius_km'], 'k-', label='Reconstructed')
plt.scatter(mr_observations['mass_solar'], mr_observations['radius_km'], c='red', label='Observations')
plt.xlabel('Mass (solar masses)')
plt.ylabel('Radius (km)')
plt.title('White-Dwarf Mass-Radius Reconstruction')
plt.legend()
plt.savefig('results/reconstruction_plot.png')
```