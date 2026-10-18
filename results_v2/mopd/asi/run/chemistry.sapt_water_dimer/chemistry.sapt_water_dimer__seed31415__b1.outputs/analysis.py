import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, lib
from scipy.linalg import solve
from scipy import sparse
import os

def read_xyz(file_path):
    with open(file_path, 'r') as f:
        data = f.read().splitlines()
    n_atoms = int(data[0])
    atoms = []
    for i in range(n_atoms):
        atoms.append(data[2+i].split())
    return atoms

def build_molecule(atoms, basis):
    mol = gto.M(atom=atoms, basis=basis, unit='Bohr')
    return mol

def run_scf(mol):
    mf = scf.RHF(mol).run()
    return mf

def compute_elst(mf_A, mf_B, mol_AB):
    # Build one-particle densities
    D_A = 2 * mf_A.mo_coeff[:, :mf_A.mol.nao] @ mf_A.mo_coeff[:, :mf_A.mol.nao].T
    D_B = 2 * mf_B.mo_coeff[:, :mf_B.mol.nao] @ mf_B.mo_coeff[:, :mf_B.mol.nao].T
    
    # Nuclear attraction between monomers
    V_nuc_AB = mf_A.mol.intor('int1e_nuc') + mf_B.mol.intor('int1e_nuc')
    V_nuc_AB = V_nuc_AB[mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao]
    
    # Coulomb matrix
    J = mf_A.mol.intor('int2e')
    J = J[mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao]
    J = J.reshape((mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao))
    
    # Compute E_elst
    E_elst = np.trace(D_A @ J[0, 0]) + np.trace(D_B @ J[0, 0])
    E_elst += np.trace(D_A @ J[0, 1]) + np.trace(D_B @ J[1, 0])
    E_elst += np.trace(D_A @ J[1, 1])
    E_elst += np.trace(V_nuc_AB)
    return E_elst

def compute_exch(mf_A, mf_B, mol_AB):
    # Build overlap matrix
    S_AB = mf_A.mol.intor('int1e_ovlp')
    S_AB = S_AB[mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao]
    
    # Build exchange matrix
    J = mf_A.mol.intor('int2e')
    J = J[mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao]
    J = J.reshape((mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao))
    
    # Compute E_exch
    E_exch = -np.trace(S_AB @ (mf_A.mo_energy[:mf_A.mol.nao] + mf_B.mo_energy[:mf_B.mol.nao] - mf_A.mo_energy[mf_A.mol.nao:] - mf_B.mo_energy[mf_B.mol.nao:]))
    E_exch += -np.trace(S_AB @ (J[0, 1] + J[1, 0]))
    return E_exch

def compute_ind(mf_A, mf_B, mol_AB):
    # Build Fock matrices
    F_A = mf_A.get_fock()
    F_B = mf_B.get_fock()
    
    # Build CPHF kernel
    def cphf_kernel(t, v):
        return F_A - F_B
    
    # Solve CPHF equations
    t_A = solve(cphf_kernel, np.zeros(mf_A.mo_coeff.shape[1]))
    t_B = solve(cphf_kernel, np.zeros(mf_B.mo_coeff.shape[1]))
    
    # Compute E_ind
    E_ind = -2 * np.sum((mf_A.mo_energy[:mf_A.mol.nao] - mf_A.mo_energy[mf_A.mol.nao:]) * t_A**2)
    E_ind -= 2 * np.sum((mf_B.mo_energy[:mf_B.mol.nao] - mf_B.mo_energy[mf_B.mol.nao:]) * t_B**2)
    return E_ind

def compute_disp(mf_A, mf_B, mol_AB):
    # Build four-center integrals
    ints = ao2mo.restore(1, mf_A.mol.intor('int2e'), mf_A.mol.nao)
    ints = ints[mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao, mf_A.mol.nao: mf_A.mol.nao + mf_B.mol.nao]
    ints = ints.reshape((mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao, mf_B.mol.nao))
    
    # Compute E_disp
    E_disp = -4 * np.sum((ints**2) / (mf_A.mo_energy[:mf_A.mol.nao] + mf_B.mo_energy[:mf_B.mol.nao] - mf_A.mo_energy[mf_A.mol.nao:] - mf_B.mo_energy[mf_B.mol.nao:]))
    return E_disp

def main():
    config = np.load('data/config.json', allow_pickle=True).item()
    basis = config['basis']
    scan_distances = config['scan_distances']
    
    # Read geometries
    dimer = read_xyz('data/dimer.xyz')
    scan_geometries = [read_xyz('data/scan_geometries.xyz')[i] for i in range(7)]
    
    # Compute components for each geometry
    results = []
    for atoms in [dimer] + scan_geometries:
        mol_AB = build_molecule(atoms, basis)
        mf_A = run_scf(mol_AB)
        mf_B = run_scf(mol_AB)
        E_elst = compute_elst(mf_A, mf_B, mol_AB)
        E_exch = compute_exch(mf_A, mf_B, mol_AB)
        E_ind = compute_ind(mf_A, mf_B, mol_AB)
        E_disp = compute_disp(mf_A, mf_B, mol_AB)
        results.append([E_elst, E_exch, E_ind, E_disp])
    
    # Save results
    np.save('results/components.npy', np.array(results[0]))
    np.save('results/distance_scan.npy', np.array(results[1:]))
    np.save('results/total_interaction.npy', np.array([sum(results[0])]))
    
    # Plot decomposition
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    components = results[0]
    plt.bar(['E_elst', 'E_exch', 'E_ind', 'E_disp'], components)
    plt.title('Component Decomposition')
    plt.subplot(1, 2, 2)
    scan_results = results[1:]
    distances = scan_distances
    plt.plot(distances, [r[0] for r in scan_results], label='E_elst')
    plt.plot(distances, [r[1] for r in scan_results], label='E_exch')
    plt.plot(distances, [r[2] for r in scan_results], label='E_ind')
    plt.plot(distances, [r[3] for r in scan_results], label='E_disp')
    plt.legend()
    plt.xlabel('O–O Distance (Å)')
    plt.ylabel('Energy (kcal/mol)')
    plt.tight_layout()
    plt.savefig('results/decomposition_plot.png')
    plt.close()

if __name__ == '__main__':
    main()