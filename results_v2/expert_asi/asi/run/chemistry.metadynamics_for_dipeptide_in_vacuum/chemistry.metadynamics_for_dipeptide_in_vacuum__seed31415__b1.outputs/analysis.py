import os
import numpy as np
import matplotlib.pyplot as plt
import openmm
from openmm import unit
from openmm.app import PDBFile, ForceField
from openmm.simulation import Simulation
from openmm.app.metadynamics import Metadynamics
from openmm import LangevinMiddleIntegrator
import matplotlib.colors as mcolors

# Define the simulation parameters
temperature = 300.0 * unit.kelvin
friction = 1.0 / unit.picosecond
timestep = 2.0 * unit.femtosecond
steps_per_deposition = 1000
simulation_steps = 10_000_000
total_steps = 10_000_000

# Load the system and force field
pdb = PDBFile('data/dipeptide.pdb')
forcefield = ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

# Define the collective variables
phi = CustomTorsionForce("theta")
phi.addTorsion(4, 6, 8, 15)
psi = CustomTorsionForce("theta")
psi.addTorsion(6, 8, 15, 17)
system.addForce(phi)
system.addForce(psi)

# Create the simulation context
platform = openmm.Platform.getPlatformByName('Reference')
simulation = Simulation(pdb.topology, system, LangevinMiddleIntegrator(temperature, friction, timestep), platform)
simulation.context.setPositions(pdb.positions)

# Minimize energy
simulation.minimizeEnergy()

# Setup metadynamics bias
meta = Metadynamics(
    system=system,
    variables=[
        ('phi', phi, -np.pi, np.pi),
        ('psi', psi, -np.pi, np.pi)
    ],
    bias_factor=8,
    height=1.0 * unit.kilojoule_per_mole,
    width=0.35,
    deposition_interval=steps_per_deposition,
    periodic=True
)

# Step the simulation and record data
bias_data = []

for step in range(0, total_steps, steps_per_deposition):
    meta.step(simulation, steps_per_deposition)
    state = simulation.context.getState(energy=True, groups={meta._force.getForceGroup()})
    bias_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    cv_values = meta.getCollectiveVariables(simulation)
    bias_data.append((cv_values[0], cv_values[1], bias_energy))

# Reconstruct the FES
phi_grid = np.linspace(-np.pi, np.pi, 72)
psi_grid = np.linspace(-np.pi, np.pi, 72)
fes = np.zeros((72, 72))
for phi_k, psi_k, h_k in bias_data:
    for i, phi in enumerate(phi_grid):
        for j, psi in enumerate(psi_grid):
            d_phi = (phi - phi_k + np.pi) % (2 * np.pi) - np.pi
            d_psi = (psi - psi_k + np.pi) % (2 * np.pi) - np.pi
            fes[i, j] += h_k * np.exp(-(d_phi**2 + d_psi**2) / (2 * 0.35**2))

# Convert to free energy and shift
gamma = 8
fes = -gamma / (gamma - 1) * fes
fes -= np.min(fes)
fes[fes > 100] = 100

# Save the FES grid
np.save('results/fes_2d.npy', fes)

# Smooth and identify minima
from scipy.ndimage import gaussian_filter
smoothed_fes = gaussian_filter(fes, sigma=2)
minima = []
for i in range(72):
    for j in range(72):
        if smoothed_fes[i, j] < np.max(smoothed_fes) - 2:
            minima.append((i, j, smoothed_fes[i, j]))

minima.sort(key=lambda x: x[2])
minima = minima[:5]

# Save minima data
with open('results/minima.csv', 'w') as f:
    for idx, (i, j, energy) in enumerate(minima):
        phi_deg = np.degrees(phi_grid[i])
        psi_deg = np.degrees(psi_grid[j])
        f.write(f'{idx+1},{phi_deg},{psi_deg},{energy}\n')

# Plot the FES
plt.figure(figsize=(8, 8))
plt.contourf(np.degrees(phi_grid), np.degrees(psi_grid), fes, levels=50, cmap='viridis')
plt.colorbar()
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.xlim(-180, 180)
plt.ylim(-180, 180)
plt.title('Free Energy Surface')
plt.savefig('results/fes_plot.png')

# Save CV definition
with open('results/cv_definition.txt', 'w') as f:
    f.write('Phi: Dihedral angle between C(ACE)-N(X)-CA(X)-C(X)\n')
    f.write('Psi: Dihedral angle between N(X)-CA(X)-C(X)-N(NME)\n')
    f.write('Atoms used: 0-based indices 4,6,8,15 for phi; 6,8,15,17 for psi\n')
    f.write('Selection rationale: Backbone dihedral angles that define the dipeptide conformation\n')