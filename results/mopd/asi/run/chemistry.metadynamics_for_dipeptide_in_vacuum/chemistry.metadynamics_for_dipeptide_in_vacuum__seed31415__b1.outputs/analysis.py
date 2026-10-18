import os
import numpy as np
import matplotlib.pyplot as plt
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import openmm.vec3 as vec3
from openmm.app import PDBFile, Simulation, Topology
from openmm import LangevinMiddleIntegrator, Platform, System
from openmm.app.metadynamics import Metadynamics, BiasVariable
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import pandas as pd

# Define the path to the PDB file
pdb_path = 'data/dipeptide.pdb'
pdb = PDBFile(pdb_path)
topology = pdb.topology
positions = pdb.positions

# Create the system
forcefield = app.ForceField('amber14-all.xml')
system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff)
# Define the collective variables
phi_torsion = app.CustomTorsionForce("theta")
phi_torsion.addTorsion(4, 6, 8, 15)
psi_torsion = app.CustomTorsionForce("theta")
psi_torsion.addTorsion(6, 8, 15, 17)
system.addForce(phi_torsion)
system.addForce(psi_torsion)

# Set up the simulation
platform = Platform.getPlatformByName('Reference')
integrator = LangevinMiddleIntegrator(300.0*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)
simulation = Simulation(topology, system, integrator, platform)
simulation.context.setPositions(positions)

# Energy minimization
simulation.minimizeEnergy()

# Define metadynamics
bias_variable = BiasVariable()
bias_variable.addVariable(phi_torsion, 0.35*unit.radians, -np.pi*unit.radians, np.pi*unit.radians)
bias_variable.addVariable(psi_torsion, 0.35*unit.radians, -np.pi*unit.radians, np.pi*unit.radians)
metadynamics = Metadynamics(simulation, [bias_variable], 1.0*unit.kilojoules_per_mole, 8.0)
# Run the simulation
simulation.step(10000000)
# Collect the bias data
bias_data = []
for i in range(0, 10000000, 1000):
    simulation.step(1000)
    # Get the current CV values
    cv_values = simulation.context.getState(getEnergy=True, groups={metadynamics._force.getForceGroup()}).getPotentialEnergy()
    # Get the current bias energy
    bias_energy = simulation.context.getState(energy=True, groups={metadynamics._force.getForceGroup()}).getPotentialEnergy()
    bias_energy = bias_energy.value_in_unit(unit.kilojoules_per_mole)
    # Append the data
    bias_data.append((cv_values, bias_energy))
bias_data = np.array(bias_data)

# Reconstruct the FES
# Define the grid
grid_size = 72
phi_grid = np.linspace(-np.pi, np.pi, grid_size)
psi_grid = np.linspace(-np.pi, np.pi, grid_size)
phi_grid_deg = np.degrees(phi_grid)
psi_grid_deg = np.degrees(psi_grid)
# Create a meshgrid
phi_mesh, psi_mesh = np.meshgrid(phi_grid, psi_grid)
# Initialize the FES
fes = np.zeros((grid_size, grid_size))
# For each deposited Gaussian
for i in range(len(bias_data)):
    # Get the center and height
    phi_center = bias_data[i][0][0]
    psi_center = bias_data[i][0][1]
    h_k = bias_data[i][1]
    # Compute the distance from the center
    d_phi = np.abs(phi_mesh - phi_center)
    d_psi = np.abs(psi_mesh - psi_center)
    # Apply periodicity
    d_phi = np.minimum(d_phi, 2*np.pi - d_phi)
    d_psi = np.minimum(d_psi, 2*np.pi - d_psi)
    # Compute the Gaussian
    gaussian = np.exp(-(d_phi**2 + d_psi**2)/(2*(0.35**2)))
    # Add the contribution to the FES
    fes += h_k * gaussian
# Convert to free energy
gamma = 8.0
fes = -gamma/(gamma - 1) * fes
# Shift the FES so the global minimum is 0
fes -= np.min(fes)
# Cap high-energy regions
fes[fes > 100] = 100
# Save the FES to a numpy file
np.save('results/fes_2d.npy', fes)

# Find the minima
# Apply Gaussian smoothing
fes_smoothed = gaussian_filter(fes, sigma=2)
# Find local minima
minima = np.argmin(fes_smoothed, axis=None)
# Convert to coordinates
minima = np.unravel_index(minima, fes_smoothed.shape)
minima = np.array(minima).T
# Filter out shallow minima
minima = minima[fes_smoothed[minima[:,0], minima[:,1]] < 2]
# Convert to degrees
minima_deg = np.degrees(minima)
# Sort by free energy
minima_deg = minima_deg[np.argsort(fes[minima[:,0], minima[:,1]])]
# Save the minima to a CSV file
df = pd.DataFrame(minima_deg, columns=['phi_deg', 'psi_deg'])
df['delta_g_kj_mol'] = fes[minima[:,0], minima[:,1]]
df['name'] = ['Minimum {}'.format(i+1) for i in range(len(df))]
df.to_csv('results/minima.csv', index=False)

# Plot the FES
plt.figure(figsize=(10, 10))
plt.contourf(phi_grid_deg, psi_grid_deg, fes, 50, cmap='viridis')
plt.colorbar()
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.title('Free Energy Surface')
plt.savefig('results/fes_plot.png')

# Write the CV definition
with open('results/cv_definition.txt', 'w') as f:
    f.write('Collective variables used in the simulation:\n')
    f.write('1. phi: Dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X), where X is the central residue (0-based atom indices 4, 6, 8, 15)\n')
    f.write('2. psi: Dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME), where X is the central residue (0-based atom indices 6, 8, 15, 17)\n')
    f.write('The CVs were selected to capture the backbone conformational changes of the dipeptide.\n')