```python
import os
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import openmm
import openmm.app as app
import openmm.unit as unit
from openmm.app.metadynamics import Metadynamics
from openmm import LangevinMiddleIntegrator, Platform, State

# Define paths
pdb_file = 'data/dipeptide.pdb'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# CV definitions
phi_indices = (4, 6, 8, 15)
psi_indices = (6, 8, 15, 17)

# System setup
pdb = app.PDBFile(pdb_file)
forcefield = app.ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

# Add CVs using CustomTorsionForce
phi_force = app.CustomTorsionForce("theta")
phi_force.addTorsion(*phi_indices)
psi_force = app.CustomTorsionForce("theta")
psi_force.addTorsion(*psi_indices)
system.addForce(phi_force)
system.addForce(psi_force)

# Integrator and simulation setup
integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond)
platform = Platform.getPlatformByName('Reference')
simulation = app.Simulation(pdb.topology, system, integrator, platform)
simulation.context.setState(State(enforcePeriodicBoundaryConditions=False))

# Metadynamics setup
meta = Metadynamics()
meta.addBiasVariable(phi_force, 0, 1, 1, 1.0 * unit.kilojoules_per_mole, 0.35 * unit.radian, 8.0)
meta.addBiasVariable(psi_force, 0, 1, 1, 1.0 * unit.kilojoules_per_mole, 0.35 * unit.radian, 8.0)
meta.setBiasFactor(8.0)
meta.setDepositionInterval(1000)
meta.setPeriodicity([True, True])

# Run simulation
simulation.minimizeEnergy()
simulation.step(10000)

# Simulation parameters
total_steps = 10000000
deposition_interval = 1000
steps_per_chunk = 1000
num_chunks = total_steps // steps_per_chunk

# Record data
bias_data = []

for chunk in range(num_chunks):
    simulation.step(steps_per_chunk)
    # Get current CV values
    current_cvs = meta.getCollectiveVariables(simulation)
    phi, psi = current_cvs
    # Get bias energy
    state = simulation.context.getState(energy=True, groups={meta._force.getForceGroup()})
    bias_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    # Record scaled Gaussian height
    k = 8.314e-3  # kJ/(mol·K)
    T = 300.0  # K
    delta_T = T * (meta.getBiasFactor() - 1)
    h_k = 1.0 * np.exp(-bias_energy / (k * delta_T))
    bias_data.append((phi, psi, h_k))

# Reconstruct FES
phi_grid = np.linspace(-np.pi, np.pi, 72)
psi_grid = np.linspace(-np.pi, np.pi, 72)
phi_grid_deg = np.degrees(phi_grid)
psi_grid_deg = np.degrees(psi_grid)
phi_grid_centered = (phi_grid + np.pi) % (2 * np.pi) - np.pi
psi_grid_centered = (psi_grid + np.pi) % (2 * np.pi) - np.pi

fes = np.zeros((72, 72))
for phi, psi, h in bias_data:
    phi_idx = int((phi + np.pi) % (2 * np.pi) * 72 / (2 * np.pi))
    psi_idx = int((psi + np.pi) % (2 * np.pi) * 72 / (2 * np.pi))
    d_phi = np.abs(phi_grid_centered - phi)
    d_phi = np.minimum(d_phi, 2 * np.pi - d_phi)
    d_psi = np.abs(psi_grid_centered - psi)
    d_psi = np.minimum(d_psi, 2 * np.pi - d_psi)
    fes += h * np.exp(-(d_phi**2 + d_psi**2) / (2 * 0.35**2))

# Convert to free energy
gamma = 8.0
fes = -gamma / (gamma - 1) * fes

# Shift FES to have global minimum at 0
min_energy = np.min(fes)
fes -= min_energy
fes[fes > 100] = 100

# Save FES
np.save(os.path.join(output_dir, 'fes_2d.npy'), fes)

# Smooth FES and find minima
fes_smoothed = scipy.ndimage.gaussian_filter(fes, sigma=2)
minima = []
for i in range(72):
    for j in range(72):
        if fes_smoothed[i, j] == np.min(fes_smoothed[i, j]):
            # Check if it's a local minimum
            is_local_min = True
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if i + dx < 0 or i + dx >= 72 or j + dy < 0 or j + dy >= 72:
                        continue
                    if fes_smoothed[i, j] <= fes_smoothed[i + dx, j + dy]:
                        is_local_min = False
                        break
                if not is_local_min:
                    break
            if is_local_min:
                depth = fes[i, j] - np.min(fes)
                if depth >= 2:
                    phi_deg = phi_grid_deg[i]
                    psi_deg = psi_grid_deg[j]
                    minima.append((phi_deg, psi_deg, depth))

minima.sort(key=lambda x: x[2])
minima = minima[:5]

# Save minima
with open(os.path.join(output_dir, 'minima.csv'), 'w') as f:
    f.write('name,phi_deg,psi_deg,delta_g_kj_mol\n')
    for idx, (phi_deg, psi_deg, delta_g) in enumerate(minima):
        f.write(f'minima_{idx}, {phi_deg:.2f}, {psi_deg:.2f}, {delta_g:.2f}\n')

# Plot FES
plt.figure(figsize=(8, 8))
plt.contourf(phi_grid_deg, psi_grid_deg, fes, levels=50, cmap='viridis')
plt.colorbar(label='Free Energy (kJ/mol)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.title('2D Free Energy Surface')
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.savefig(os.path.join(output_dir, 'fes_plot.png'))

# Save CV definition
with open(os.path.join(output_dir, 'cv_definition.txt'), 'w') as f:
    f.write("Collective Variables for Dipeptide:\n")
    f.write("1. Phi (φ): Dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X), indices (4, 6, 8, 15)\n")
    f.write("2. Psi (ψ): Dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME), indices (6, 8, 15, 17)\n")
    f.write("These CVs capture the backbone dihedral angles of the central residue, which are critical for conformational sampling in dipeptides.")
```