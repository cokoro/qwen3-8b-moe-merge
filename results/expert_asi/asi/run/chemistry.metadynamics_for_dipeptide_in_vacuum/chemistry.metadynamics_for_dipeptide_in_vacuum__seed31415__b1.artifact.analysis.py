import os
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
from openmm import unit, LangevinMiddleIntegrator, CustomTorsionForce
from openmm.app.metadynamics import Metadynamics, BiasVariable
from scipy.ndimage import gaussian_filter

# Constants
k_B = 8.314e-3  # kJ/mol/K
T = 300.0  # K
gamma = 8.0
h0 = 1.0  # kJ/mol
sigma = 0.35  # rad
deposition_interval = 1000
steps = 10_000_000  # 20 ns
grid_size = 72
bin_width_deg = 5.0  # 5 degrees per bin
bin_width_rad = np.deg2rad(bin_width_deg)

# Read initial structure
pdb = app.PDBFile('data/dipeptide.pdb')
system = pdb.topology.createSystem(pdb.positions, nonbondedMethod=app.NoCutoff,
                                   forceField='amber14-all.xml')
integrator = LangevinMiddleIntegrator(T, 1.0/unit.picosecond, 2.0*unit.femtosecond)
simulation = app.Simulation(pdb.topology, system, integrator, platform=mm.Platform.getPlatformByName('Reference'))
simulation.context.setPositions(pdb.positions)

# Define collective variables
phi_force = CustomTorsionForce('theta')
phi_force.addTorsion(4, 6, 8, 15)
psi_force = CustomTorsionForce('theta')
psi_force.addTorsion(6, 8, 15, 17)
phi_force.setForceGroup(1)
psi_force.setForceGroup(2)
system.addForce(phi_force)
system.addForce(psi_force)

# Initialize metadynamics
meta = Metadynamics(system, [BiasVariable('phi', 0.0, -np.pi, np.pi, 1.0, 0.35, gamma, 0.0, 0.0, 0.0, 0.0)])
meta.addBiasVariable('psi', 0.0, -np.pi, np.pi, 1.0, 0.35, gamma, 0.0, 0.0, 0.0, 0.0)
meta.setDepositionInterval(deposition_interval)

# Run simulation
simulation.step(steps)

# Collect data
bias_energy = []
phi_values = []
psi_values = []
for _ in range(1000):  # 1000 steps per chunk
    simulation.step(deposition_interval)
    state = simulation.context.getState(getEnergy=True, groups={meta._force.getForceGroup()})
    bias_energy.append(state.getPotentialEnergy().value_in_unit(kilojoules_per_mole))
    phi, psi = meta.getCollectiveVariables(simulation)
    phi_values.append(phi)
    psi_values.append(psi)

# Reconstruct FES
phi_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
psi_grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
grid_phi, grid_psi = np.meshgrid(phi_grid, psi_grid, indexing='ij')
fes = np.zeros_like(grid_phi)

for i in range(len(phi_values)):
    phi = phi_values[i]
    psi = psi_values[i]
    energy = bias_energy[i]
    # Compute h_k
    delta_t = T * (gamma - 1)
    h_k = h0 * np.exp(-energy / (k_B * delta_t))
    # Compute Gaussian contribution
    d_phi = np.abs(phi_grid - phi) % (2 * np.pi) - np.pi
    d_psi = np.abs(psi_grid - psi) % (2 * np.pi) - np.pi
    fes += h_k * np.exp(-(d_phi**2 + d_psi**2) / (2 * sigma**2))

# Convert to free energy
fes *= -gamma / (gamma - 1)
# Shift to zero baseline
fes -= np.min(fes)
# Cap high-energy regions
fes[fes > 100.0] = 100.0

# Save FES
np.save('results/fes_2d.npy', fes)

# Find minima
smoothed = gaussian_filter(fes, sigma=2)
minima = []
for i in range(grid_size):
    for j in range(grid_size):
        if smoothed[i,j] > np.max(smoothed) - 2.0 and fes[i,j] < np.min(fes) + 2.0:
            phi_deg = np.degrees(phi_grid[i])
            psi_deg = np.degrees(psi_grid[j])
            delta_g = fes[i,j]
            minima.append((phi_deg, psi_deg, delta_g))

minima.sort(key=lambda x: x[2])
minima = minima[:5]

# Save minima
np.savetxt('results/minima.csv', minima, header='phi_deg,psi_deg,delta_g_kj_mol', fmt='%.1f,%.1f,%.1f')

# Plot FES
plt.figure(figsize=(8, 8))
plt.imshow(fes, extent=[-180, 180, -180, 180], cmap='viridis', origin='lower')
plt.colorbar(label='Free energy (kJ/mol)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.title('2D Free Energy Surface')
plt.savefig('results/fes_plot.png')

# CV definition
with open('results/cv_definition.txt', 'w') as f:
    f.write('phi: C(ACE)-N(X)-CA(X)-C(X) torsion (atoms 4,6,8,15)\n')
    f.write('psi: N(X)-CA(X)-C(X)-N(NME) torsion (atoms 6,8,15,17)\n')
    f.write('Selection rationale: Backbone dihedral angles that capture the conformational space of a dipeptide.')