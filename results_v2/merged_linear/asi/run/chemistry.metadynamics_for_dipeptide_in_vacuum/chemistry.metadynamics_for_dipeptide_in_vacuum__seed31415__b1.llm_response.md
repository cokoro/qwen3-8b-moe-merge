```python
import numpy as np
import openmm as mm
from openmm import unit
from openmm.app import PDBFile, Topology
from openmm.unit import kilojoules_per_mole, nanometers, picoseconds, seconds, kelvin
from openmm.app import Simulation
from openmm import LangevinMiddleIntegrator
from openmm.app import ForceField
from openmm.app import CustomTorsionForce
from openmm.app import NoCutoff
from openmm.app.metadynamics import Metadynamics, BiasVariable
from openmm import ReferencePlatform
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.cm as cm
import os
import csv
from matplotlib.ticker import NullFormatter

# Set the working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the system
pdb = PDBFile('data/dipeptide.pdb')
topology = pdb.topology
positions = pdb.positions

# Define force field
forcefield = ForceField('amber14-all.xml')

# Create the system
system = forcefield.createSystem(topology, nonbondedMethod=NoCutoff, constraints=None, rigidWater=False)

# Define the collective variables
phi_force = CustomTorsionForce("theta")
phi_force.addTorsion(4, 6, 8, 15)
phi_force.addTorsion(4, 6, 8, 15)
system.addForce(phi_force)
phi_force.setForceGroup(1)

psi_force = CustomTorsionForce("theta")
psi_force.addTorsion(6, 8, 15, 17)
psi_force.addTorsion(6, 8, 15, 17)
system.addForce(psi_force)
psi_force.setForceGroup(2)

# Define the integrator
temperature = 300.0 * kelvin
friction = 1.0 / picoseconds
timestep = 2.0 * picoseconds
integrator = LangevinMiddleIntegrator(temperature, friction, timestep)

# Create the simulation
simulation = Simulation(topology, system, integrator, ReferencePlatform())

# Set initial positions
simulation.context.setPositions(positions)

# Minimize energy
simulation.minimizeEnergy()

# Define the metadynamics protocol
bias_height = 1.0 * kilojoules_per_mole
sigma = 0.35
gamma = 8.0
deposition_interval = 1000
metadynamics = Metadynamics(simulation, BiasVariable(phi_force, "phi", -np.pi, np.pi), BiasVariable(psi_force, "psi", -np.pi, np.pi), bias_height, sigma, gamma, deposition_interval)

# Run the simulation
num_steps = 10000000
metadynamics.step(simulation, num_steps)

# Read the simulation data
data = []
for _ in range(num_steps // deposition_interval):
    state = simulation.context.getState(energy=True, groups={phi_force.getForceGroup(), psi_force.getForceGroup()})
    phi, psi = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole), state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    data.append((phi, psi))

data = np.array(data)

# Reconstruct the FES
grid = np.linspace(-np.pi, np.pi, 72)
phi_grid, psi_grid = np.meshgrid(grid, grid)
phi_flat = phi_grid.flatten()
psi_flat = psi_grid.flatten()
fes = np.zeros((72, 72))

for i in range(len(data)):
    phi_k, psi_k = data[i]
    dist_phi = np.abs(phi_flat - phi_k)
    dist_phi = np.where(dist_phi > np.pi, 2 * np.pi - dist_phi, dist_phi)
    dist_psi = np.abs(psi_flat - psi_k)
    dist_psi = np.where(dist_psi > np.pi, 2 * np.pi - dist_psi, dist_psi)
    fes += np.exp(-(dist_phi ** 2 + dist_psi ** 2) / (2 * sigma ** 2)) * bias_height

# Convert to free energy
fes *= -gamma / (gamma - 1)
fes -= fes.min()

# Cap high-energy regions
fes = np.where(fes > 100.0, 100.0, fes)

# Save the FES
np.save('results/fes_2d.npy', fes)

# Identify minima
fes_smoothed = np.zeros_like(fes)
for i in range(72):
    for j in range(72):
        sum_val = 0.0
        count = 0
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ni, nj = i + di, j + dj
                ni = np.clip(ni, 0, 71)
                nj = np.clip(nj, 0, 71)
                sum_val += fes[ni, nj]
                count += 1
        fes_smoothed[i, j] = sum_val / count

minima = []
for i in range(72):
    for j in range(72):
        if i < 2 or i > 69 or j < 2 or j > 69:
            continue
        neighbors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ni, nj = i + di, j + dj
                ni = np.clip(ni, 0, 71)
                nj = np.clip(nj, 0, 71)
                neighbors.append(fes[ni, nj])
        if min(fes[i, j], min(neighbors)) == fes[i, j]:
            if fes[i, j] < 2.0:
                continue
            minima.append((i, j))

minima = np.array(minima)
minima.sort()
minima = minima[:5]

# Save the minima
with open('results/minima.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'phi_deg', 'psi_deg', 'delta_g_kj_mol'])
    for i in range(len(minima)):
        name = f"minima_{i+1}"
        phi_deg = (minima[i][0] * (180 / 72) - 180) % 360
        psi_deg = (minima[i][1] * (180 / 72) - 180) % 360
        delta_g_kj_mol = fes[minima[i][0], minima[i][1]]
        writer.writerow([name, phi_deg, psi_deg, delta_g_kj_mol])

# Plot the FES
plt.figure(figsize=(8, 8))
plt.imshow(fes, extent=[-180, 180, -180, 180], cmap='viridis', origin='lower')
plt.colorbar()
plt.xlabel('phi (degrees)')
plt.ylabel('psi (degrees)')
plt.title('Free Energy Surface of Cysteine Dipeptide')
plt.savefig('results/fes_plot.png')

# Save the CV definition
with open('results/cv_definition.txt', 'w') as f:
    f.write('phi: Dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X) where X is the central residue, 0-based atom indices (4, 6, 8, 15)\n')
    f.write('psi: Dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME) where X is the central residue, 0-based atom indices (6, 8, 15, 17)')
```