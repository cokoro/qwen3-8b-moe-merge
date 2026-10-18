import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmm.vec3 import Vec3
from openmm.metadynamics import Metadynamics

# Paths
pdb_file = 'data/dipeptide.pdb'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# Load system
pdb = PDBFile(pdb_file)
forcefield = ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=HBonds)

# Add custom torsion forces for CVs
phi_force = CustomTorsionForce("theta")
phi_force.addTorsion(4, 6, 8, 15)
psi_force = CustomTorsionForce("theta")
psi_force.addTorsion(6, 8, 15, 17)
system.addForce(phi_force)
system.addForce(psi_force)

# Minimize energy
integrator = LangevinMiddleIntegrator(300.0 * kelvin, 1.0 / picosecond, 2.0 * femtosecond)
platform = Platform.getPlatformByName("Reference")
simulation = Simulation(pdb.topology, system, integrator, platform)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()

# Setup metadynamics
meta = Metadynamics()
meta.addCollectiveVariable('phi', phi_force)
meta.addCollectiveVariable('psi', psi_force)
meta.setGaussianHeight(1.0 * kilojoule_per_mole)
meta.setGaussianWidth(0.35 * radians)
meta.setBiasFactor(8.0)
meta.setDepositionInterval(1000)

# Simulation parameters
n_steps = 10_000_000
chunk_size = 1000

# Run simulation
simulation.context.setState(simulation.context.getState(groups={phi_force.getForceGroup(), psi_force.getForceGroup()}))
bias_energy = 0.0 * kilojoule_per_mole
deposited = []

for i in range(n_steps // chunk_size):
    meta.step(simulation, chunk_size)
    # Record CV values and bias energy
    cv = meta.getCollectiveVariables(simulation)
    state = simulation.context.getState(energy=True, groups={meta._force.getForceGroup()})
    bias_energy = state.getPotentialEnergy()
    deposited.append((cv[0], cv[1], bias_energy))

# Save deposited data
np.savetxt(os.path.join(output_dir, 'deposited.txt'), deposited, delimiter=' ', header='phi psi bias_energy')

# Reconstruct FES
phi_grid = np.linspace(-np.pi, np.pi, 72)
psi_grid = np.linspace(-np.pi, np.pi, 72)
phi_grid_deg = np.degrees(phi_grid)
psi_grid_deg = np.degrees(psi_grid)

fes = np.zeros((72, 72))
for i, (phi, psi, bias_energy) in enumerate(deposited):
    phi_idx = np.argmin(np.abs(phi_grid - phi))
    psi_idx = np.argmin(np.abs(psi_grid - psi))
    h_k = 1.0 * kilojoule_per_mole * np.exp(-bias_energy / (8.314e-3 * kelvin * (8.0 - 1.0)))
    dx = (phi - phi_grid[phi_idx]) % (2 * np.pi)
    dy = (psi - psi_grid[psi_idx]) % (2 * np.pi)
    fes[phi_idx, psi_idx] += h_k * np.exp(-(dx**2 + dy**2) / (2 * (0.35 * radians)**2))

# Convert to free energy
fes = -8.0 / (8.0 - 1.0) * fes
fes = np.ma.masked_where(fes > 100 * kilojoule_per_mole, fes)
fes = fes - fes.min()

# Save FES
np.save(os.path.join(output_dir, 'fes_2d.npy'), fes)

# Smoothing and minima detection
fes_smooth = gaussian_filter(fes, sigma=2)
minima = []
for i in range(72):
    for j in range(72):
        if fes_smooth[i, j] > fes_smooth[i-1, j] and fes_smooth[i, j] > fes_smooth[i+1, j] and fes_smooth[i, j] > fes_smooth[i, j-1] and fes_smooth[i, j] > fes_smooth[i, j+1]:
            if fes[i, j] > 2 * kilojoule_per_mole:
                minima.append((i, j))

minima.sort(key=lambda x: fes[x[0], x[1]])
minima = minima[:5]

# Save minima
with open(os.path.join(output_dir, 'minima.csv'), 'w') as f:
    for i, (x, y) in enumerate(minima):
        f.write(f'Minima_{i+1}, {np.degrees(phi_grid[x])}, {np.degrees(psi_grid[y])}, {fes[x, y].value_in_unit(kilojoule_per_mole)}\n')

# Plot FES
plt.figure(figsize=(8, 6))
plt.imshow(fes, extent=[-180, 180, -180, 180], aspect='auto', cmap='viridis')
plt.colorbar(label='Free energy (kJ/mol)')
plt.xlabel('Psi (degrees)')
plt.ylabel('Phi (degrees)')
plt.title('Free energy surface of cysteine dipeptide')
plt.savefig(os.path.join(output_dir, 'fes_plot.png'))
plt.close()

# Save CV definition
with open(os.path.join(output_dir, 'cv_definition.txt'), 'w') as f:
    f.write('Collective variables used in the simulation:\n')
    f.write('1. phi (phi): Dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X) where X is the central residue, 0-based atom indices (4, 6, 8, 15)\n')
    f.write('2. psi (psi): Dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME) where X is the central residue, 0-based atom indices (6, 8, 15, 17)\n')
    f.write('Rationale: These two dihedral angles define the backbone conformation of the dipeptide and are commonly used to characterize the conformational space of peptides.\n')