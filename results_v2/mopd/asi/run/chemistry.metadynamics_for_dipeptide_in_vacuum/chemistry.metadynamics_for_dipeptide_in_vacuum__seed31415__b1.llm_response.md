```python
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
import csv
import os

# Setup simulation
pdb = app.PDBFile('data/dipeptide.pdb')
forcefield = app.ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
integrator = mm.LangevinMiddleIntegrator(300.0*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtoseconds)
simulation = app.Simulation(pdb.topology, system, integrator, app.ReferencePlatform())
simulation.context.setPositions(pdb.positions)

# Energy minimization
simulation.minimizeEnergy()

# Define collective variables
phi_torsion = app.CustomTorsionForce("theta")
phi_torsion.addTorsion(4, 6, 8, 15)
psi_torsion = app.CustomTorsionForce("theta")
psi_torsion.addTorsion(6, 8, 15, 17)
system.addForce(phi_torsion)
system.addForce(psi_torsion)
phi_force = phi_torsion
psi_force = psi_torsion

# Set up metadynamics
meta = app.Metadynamics(system, [phi_torsion, psi_torsion], periodicity=[True, True], gaussianWidth=0.35*unit.radians, biasFactor=8.0, initialHeight=1.0*unit.kilojoules_per_mole, depositionInterval=1000)
simulation.addForce(meta)

# Run simulation
simulation.step(10000000)

# Collect data
data = []
for i in range(0, 10000000, 1000):
    simulation.step(1000)
    state = simulation.context.getState(getEnergy=True, groups={meta._force.getForceGroup()})
    bias_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    current_cvs = meta.getCollectiveVariables(simulation)
    data.append((current_cvs[0], current_cvs[1], bias_energy))

# Reconstruct FES
phi_grid = np.linspace(-np.pi, np.pi, 72)
psi_grid = np.linspace(-np.pi, np.pi, 72)
phi_grid_deg = np.degrees(phi_grid)
psi_grid_deg = np.degrees(psi_grid)
phi_mesh, psi_mesh = np.meshgrid(phi_grid, psi_grid)
phi_mesh_deg, psi_mesh_deg = np.degrees(phi_mesh), np.degrees(psi_mesh)
phi_mesh_rad, psi_mesh_rad = phi_mesh, psi_mesh

fes = np.zeros((72, 72))
for phi, psi, height in data:
    phi_idx = np.argmin(np.abs(phi_grid - phi))
    psi_idx = np.argmin(np.abs(psi_grid - psi))
    dist_phi = np.abs(phi_grid - phi)
    dist_phi[dist_phi > np.pi] -= 2*np.pi
    dist_phi[dist_phi < -np.pi] += 2*np.pi
    dist_psi = np.abs(psi_grid - psi)
    dist_psi[dist_psi > np.pi] -= 2*np.pi
    dist_psi[dist_psi < -np.pi] += 2*np.pi
    fes[phi_idx, psi_idx] += height * np.exp(-((dist_phi**2 + dist_psi**2)/(2*0.35**2)))

# Convert to free energy
gamma = 8.0
fes = -gamma/(gamma-1) * fes
fes -= np.min(fes)
fes[fes > 100] = 100

# Apply Gaussian smoothing
fes = gaussian_filter(fes, sigma=2)

# Find minima
minima = []
for i in range(72):
    for j in range(72):
        if fes[i,j] < np.min(fes) + 2 and fes[i,j] < np.max(fes) - 2:
            minima.append((phi_grid[i], psi_grid[j], fes[i,j]))
minima.sort(key=lambda x: x[2])
minima = minima[:5]

# Save results
np.save('results/fes_2d.npy', fes)
with open('results/minima.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['name', 'phi_deg', 'psi_deg', 'delta_g_kj_mol'])
    for i, (phi, psi, g) in enumerate(minima):
        writer.writerow([f'minima_{i}', np.degrees(phi), np.degrees(psi), g])
plt.figure(figsize=(8,8))
plt.contourf(phi_mesh_deg, psi_mesh_deg, fes, levels=100, cmap='viridis')
plt.colorbar()
plt.xlabel('phi (degrees)')
plt.ylabel('psi (degrees)')
plt.title('Free Energy Surface')
plt.savefig('results/fes_plot.png')
with open('results/cv_definition.txt', 'w') as f:
    f.write("Collective variables defined as dihedral angles:\n")
    f.write("phi (φ): C(ACE)-N(X)-CA(X)-C(X) with indices (4,6,8,15)\n")
    f.write("psi (ψ): N(X)-CA(X)-C(X)-N(NME) with indices (6,8,15,17)\n")
    f.write("Selected for capturing backbone conformational changes in dipeptide\n")
```