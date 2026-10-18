```bash
#!<abs_path>
# Generate the analysis script and run the simulation
cat > analysis.py << 'EOF'
from openmm import LangevinMiddleIntegrator, NoCutoff, Platform, unit
from openmm.app import PDBFile, Simulation, ForceField
from openmm.app.metadynamics import Metadynamics
import numpy as np
import matplotlib.pyplot as plt

# --- System setup ---
# Load the topology and initial coordinates
pdb = PDBFile('data/dipeptide.pdb')
forcefield = ForceField('amber14-all.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None)
integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0/unit.picosecond, 2.0/unit.femtosecond)
simulation = Simulation(pdb.topology, system, integrator)
simulation.loadState('data/dipeptide.chk')

# --- Collective variables ---
phi = CustomTorsionForce('theta')
phi.addTorsion(4, 6, 8, 15)
psi = CustomTorsionForce('theta')
psi.addTorsion(6, 8, 15, 17)
phi.setUsesPeriodicBoundaryConditions(True)
psi.setUsesPeriodicBoundaryConditions(True)
simulation.context.setPositions(pdb.positions)
simulation.context.setVelocitiesToTemperature(300.0 * unit.kelvin)

# --- Bias parameters ---
gamma = 8
h0 = 1.0 * unit.kilojoule_per_mole
sigma = 0.35 * unit.radians

# --- Metadynamics simulation ---
meta = Metadynamics(simulation.context, [phi, psi], bias_factor=gamma, h0=h0, sigma=sigma, deposition_interval=1000)
simulation.minimizeEnergy()
simulation.step(10000)

# --- Bias history collection ---
bias_hist = []

# Step the simulation in chunks of 1000 steps and record bias data
for step in range(10000000):
    if step % 1000 == 0:
        # Record the current CVs and bias energy
        state = simulation.context.getState(energy=True, groups={meta._force.getForceGroup()})
        phi_val = state.getForce(0).getEnergy()
        psi_val = state.getForce(1).getEnergy()
        bias_energy = state.getPotentialEnergy()
        bias_hist.append((phi_val, psi_val, bias_energy))
    if step % 10000 == 0:
        print(f"Step {step} of 10,000,000")
    meta.step(simulation, 1000)

# --- Bias reconstruction ---
phi_grid = np.linspace(-np.pi, np.pi, 72)
psi_grid = np.linspace(-np.pi, np.pi, 72)
fes = np.zeros((72, 72))
for k in range(len(bias_hist)):
    phi_k, psi_k, bias_energy = bias_hist[k]
    for i in range(72):
        for j in range(72):
            d_phi = phi_grid[i] - phi_k
            d_psi = psi_grid[j] - psi_k
            d_phi = d_phi % (2 * np.pi) - np.pi
            d_psi = d_psi % (2 * np.pi) - np.pi
            exp_term = np.exp(-(d_phi**2 + d_psi**2) / (2 * sigma**2))
            fes[i, j] += h0 * np.exp(-bias_energy / (8.314e-3 * 300.0 * (gamma - 1.0))) * exp_term

# --- Free energy reconstruction ---
fes -= np.min(fes)
fes[fes > 100.0] = 100.0

# --- Minima identification ---
fes_smooth = np.zeros((72, 72))
for i in range(72):
    for j in range(72):
        if i > 1 and i < 70 and j > 1 and j < 70:
            fes_smooth[i, j] = (fes[i-1, j-1] + fes[i-1, j] + fes[i-1, j+1] +
                                fes[i, j-1] + fes[i, j] + fes[i, j+1] +
                                fes[i+1, j-1] + fes[i+1, j] + fes[i+1, j+1]) / 9.0
        else:
            fes_smooth[i, j] = fes[i, j]

minima = []
for i in range(72):
    for j in range(72):
        if i > 1 and i < 70 and j > 1 and j < 70:
            if fes_smooth[i, j] < np.min([fes_smooth[i-1, j], fes_smooth[i+1, j], fes_smooth[i, j-1], fes_smooth[i, j+1]]):
                depth = fes_smooth[i, j] - np.min([fes_smooth[i-1, j], fes_smooth[i+1, j], fes_smooth[i, j-1], fes_smooth[i, j+1]])
                if depth > 2.0:
                    minima.append((i, j, depth))
minima.sort(key=lambda x: x[2])

# --- Output ---
np.save('results/fes_2d.npy', fes)
np.savetxt('results/minima.csv', minima, delimiter=',', header='name,phi_deg,psi_deg,delta_g_kj_mol')
plt.contourf(phi_grid * 180 / np.pi, psi_grid * 180 / np.pi, fes, cmap='viridis')
plt.colorbar(label='Free energy (kJ/mol)')
plt.xlabel('Phi (degrees)')
plt.ylabel('Psi (degrees)')
plt.savefig('results/fes_plot.png')
with open('results/cv_definition.txt', 'w') as f:
    f.write('phi: dihedral angle defined by atoms C(ACE)-N(X)-CA(X)-C(X), 0-based atom indices (4, 6, 8, 15)\npsi: dihedral angle defined by atoms N(X)-CA(X)-C(X)-N(NME), 0-based atom indices (6, 8, 15, 17)')
EOF
python analysis.py
```