```python
import os
import numpy as np
import matplotlib.pyplot as plt
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm import LangevinMiddleIntegrator
from openmm.app.metadynamics import Metadynamics, BiasVariable
import sys
sys.path.append('data')
sys.path.append('results')

# Define paths
pdb_file = 'data/dipeptide.pdb'
output_dir = 'results'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the PDB file
pdb = PDBFile(pdb_file)

# Define the system
system = System()
for atom in pdb.topology.atoms():
    system.addParticle(1.0 * amu)  # Assuming all atoms have the same mass for simplicity

# Define the force field
forcefield = ForceField('amber14-all.xml')
forcefield.addSystem(pdb.topology, system)

# Define collective variables
phi_indices = (4, 6, 8, 15)
psi_indices = (6, 8, 15, 17)

# Create custom torsion forces for phi and psi
phi_force = CustomTorsionForce("theta")
phi_force.addTorsion(*phi_indices)
psi_force = CustomTorsionForce("theta")
psi_force.addTorsion(*psi_indices)

# Add forces to the system
system.addForce(phi_force)
system.addForce(psi_force)

# Set up the simulation
integrator = LangevinMiddleIntegrator(300.0 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
platform = Platform.getPlatformByName('Reference')
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setState(pdb.positions)

# Minimize energy
simulation.minimizeEnergy()

# Define metadynamics parameters
gamma = 8
h0 = 1.0 * kilojoules_per_mole
sigma = 0.35 * radians
deposition_interval = 1000
temperature = 300.0 * kelvin
boltzmann_constant = 8.314e-3 * kilojoules_per_mole_per_kelvin

# Set up metadynamics
meta = Metadynamics()
meta.addBiasVariable(BiasVariable(phi_force, phi_indices, -pi, pi, 1.0, 0.0, 0.0))
meta.addBiasVariable(BiasVariable(psi_force, psi_indices, -pi, pi, 1.0, 0.0, 0.0))

# Run the simulation
simulation.context.setState(simulation.context.getState(getForces=True, groups={phi_force.getForceGroup(), psi_force.getForceGroup()}))
simulation.context.setState(simulation.context.getState(getEnergy=True))

# Record data
deposition_data = []

# Run the simulation in chunks
for step in range(10_000_000):
    if step % deposition_interval == 0 and step != 0:
        # Step the simulation
        meta.step(simulation, deposition_interval)
        
        # Get current CV values
        current_cvs = meta.getCollectiveVariables(simulation)
        phi, psi = current_cvs
        
        # Get current bias energy
        state = simulation.context.getState(energy=True, groups={phi_force.getForceGroup(), psi_force.getForceGroup()})
        bias_energy = state.getPotentialEnergy()
        bias_energy_kj = bias_energy.value_in_unit(kilojoules_per_mole)
        
        # Calculate scaled Gaussian height
        delta_t = temperature * (gamma - 1)
        h_k = h0 * np.exp(-bias_energy_kj / (boltzmann_constant * delta_t))
        
        # Record data
        deposition_data.append((phi, psi, h_k))
    
    # Step the simulation
    simulation.step(1)

# Reconstruct FES
phi_grid = np.linspace(-pi, pi, 72)
psi_grid = np.linspace(-pi, pi, 72)
phi_grid_deg = np.degrees(phi_grid)
psi_grid_deg = np.degrees(psi_grid)
phi_grid_centered = (phi_grid[:-1] + phi_grid[1:]) / 2
psi_grid_centered = (psi_grid[:-1] + psi_grid[1:]) / 2

fes = np.zeros((72, 72))
for phi, psi, h in deposition_data:
    for i in range(72):
        for j in range(72):
            d_phi = np.abs(phi_grid[i] - phi)
            d_phi = np.min([d_phi, 2 * pi - d_phi])
            d_psi = np.abs(psi_grid[j] - psi)
            d_psi = np.min([d_psi, 2 * pi - d_psi])
            fes[i, j] += h * np.exp(-(d_phi**2 + d_psi**2) / (2 * sigma**2))

# Convert to free energy
gamma = 8
fes_free_energy = -gamma / (gamma - 1) * fes

# Shift FES so the global minimum is 0
min_energy = np.min(fes_free_energy)
fes_free_energy -= min_energy

# Cap high-energy regions
fes_free_energy[fes_free_energy > 100] = 100

# Save FES
np.save(os.path.join(output_dir, 'fes_2d.npy'), fes_free_energy)

# Smooth FES
from scipy.ndimage import gaussian_filter
smoothed_fes = gaussian_filter(fes_free_energy, sigma=2)

# Find local minima
from scipy.ndimage import label, find_objects
labels, num_objects = label(smoothed_fes)
minima = []
for label_num in range(1, num_objects + 1):
    indices = np.where(labels == label_num)
    phi_idx, psi_idx = indices
    phi = phi_grid_centered[phi_idx[0]]
    psi = psi_grid_centered[psi_idx[0]]
    energy = smoothed_fes[phi_idx[0], psi_idx[0]]
    depth = energy - np.max(smoothed_fes)
    if depth < -2:
        continue
    minima.append((phi, psi, energy))

# Sort minima by energy
minima.sort(key=lambda x: x[2])
minima = minima[:5]

# Save minima
minima_data = []
for phi, psi, energy in minima:
    phi_deg = np.degrees(phi)
    psi_deg = np.degrees(psi)
    delta_g = energy - min_energy
    minima_data.append({
        'name': f'Minima {minima.index((phi, psi, energy)) + 1}',
        'phi_deg': phi_deg,
        'psi_deg': psi_deg,
        'delta_g_kj_mol': delta_g
    })

import pandas as pd
df = pd.DataFrame(minima_data)
df.to_csv(os.path.join(output_dir, 'minima.csv'), index=False)

# Plot FES
plt.figure(figsize=(8, 6))
plt.imshow(fes_free_energy, extent=[-180, 180, -180, 180], origin='lower', cmap='viridis')
plt.colorbar(label='Free Energy (kJ/mol)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.title('Free Energy Surface')
plt.savefig(os.path.join(output_dir, 'fes_plot.png'))

# Write CV definition
with open(os.path.join(output_dir, 'cv_definition.txt'), 'w') as f:
    f.write("Collective Variables Definition:\n")
    f.write("1. Phi (φ): Dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X) where X is the central residue, 0-based atom indices (4, 6, 8, 15)\n")
    f.write("2. Psi (ψ): Dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME) where X is the central residue, 0-based atom indices (6, 8, 15, 17)\n")
    f.write("Rationale: These dihedral angles capture the backbone conformational changes of the dipeptide, allowing us to explore the conformational space and compute the free energy surface.\n")
```