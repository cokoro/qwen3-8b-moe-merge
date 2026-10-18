import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, lib
import json

def read_xyz(path):
    with open(path) as f:
        lines = f.read().splitlines()
    natom = int(lines[0])
    coords = np.array([list(map(float, lines[i+2].split())) for i in range(natom)])
    symbols = [lines[i+1].strip() for i in range(natom)]
    return symbols, coords

def build_mol(symbols, coords, basis, unit='Angstrom'):
    mol = gto.M(
        atom=[(s, (x, y, z)) for s, (x, y, z) in zip(symbols, coords)],
        basis=basis,
        unit=unit,
        verbose=0
    )
    return mol

def run_scf(mol):
    mf = scf.RHF(mol).run()
    return mf

def compute_elst(mf_A, mf_B, mol_AB, basis):
    # Build V_AB_nuc
    V_nuc_AB = mol_AB.intor('int1e_nuc')
    # Build J
    J = mol_AB.intor('int2e')
    # Build D_A and D_B
    D_A = mf_A.make_rdm1()
    D_B = mf_B.make_rdm1()
    # Compute E_elst
    E_elst = np.trace(D_A @ V_nuc_AB) + np.trace(D_B @ V_nuc_AB)
    E_elst += np.trace(D_A @ J) + np.trace(D_B @ J)
    # Subtract V_NN
    V_NN = np.sum([np.dot(mol_AB.atom_coords()[i], mol_AB.atom_coords()[j]) for i in range(len(mol_AB.atom)) for j in range(len(mol_AB.atom))])
    E_elst -= V_NN
    return E_elst

def compute_exch(mf_A, mf_B, mol_AB, basis):
    # Build S_AB
    S_AB = np.dot(mf_A.mo_coeff, np.dot(mol_AB.intor('int1e_ovlp'), mf_B.mo_coeff.T))
    # Build V_AB_nuc
    V_nuc_AB = mol_AB.intor('int1e_nuc')
    # Build J
    J = mol_AB.intor('int2e')
    # Compute E_exch
    E_exch = -np.trace(S_AB @ (mf_A.get_fock() + mf_B.get_fock()))
    E_exch += np.trace(S_AB @ (V_nuc_AB + J))
    return E_exch

def compute_ind(mf_A, mf_B, mol_AB, basis):
    # Build V_AB_nuc
    V_nuc_AB = mol_AB.intor('int1e_nuc')
    # Build J
    J = mol_AB.intor('int2e')
    # Build CPHF
    cphf = scf.cphf.CPHF(mf_A)
    t = cphf.solve(V_nuc_AB, mf_A.get_fock(), mf_B.mo_coeff)
    E_ind = -2 * np.sum((mf_A.mo_energy - mf_B.mo_energy) * t**2)
    return E_ind

def compute_disp(mf_A, mf_B, mol_AB, basis):
    # Build (ia|jb)
    ints = mol_AB.intor('int2e')
    # Build E_disp
    E_disp = -4 * np.sum(np.abs(ints)**2 / (mf_A.mo_energy + mf_B.mo_energy - mf_A.mo_energy - mf_B.mo_energy))
    return E_disp

def main():
    # Read input
    with open('data/config.json') as f:
        config = json.load(f)
    symbols, coords = read_xyz('data/dimer.xyz')
    basis = config['basis']
    scan_distances = config['scan_distances']
    
    # Build reference geometry
    mol_AB = build_mol(symbols, coords, basis)
    mf_A = run_scf(build_mol(symbols[:2], coords[:2], basis))
    mf_B = run_scf(build_mol(symbols[2:], coords[2:], basis))
    
    # Compute components
    E_elst = compute_elst(mf_A, mf_B, mol_AB, basis)
    E_exch = compute_exch(mf_A, mf_B, mol_AB, basis)
    E_ind = compute_ind(mf_A, mf_B, mol_AB, basis)
    E_disp = compute_disp(mf_A, mf_B, mol_AB, basis)
    
    # Save components
    np.save('results/components.npy', np.array([E_elst, E_exch, E_ind, E_disp]))
    np.save('results/total_interaction.npy', E_elst + E_exch + E_ind + E_disp)
    
    # Read scan geometries
    scan_coords = []
    with open('data/scan_geometries.xyz') as f:
        lines = f.read().splitlines()
    for i in range(7):
        natom = int(lines[i*2])
        coords = np.array([list(map(float, lines[i*2+2+2*j].split())) for j in range(natom)])
        scan_coords.append(coords)
    
    # Compute scan
    scan_components = []
    for coords in scan_coords:
        mol_AB = build_mol(symbols, coords, basis)
        mf_A = run_scf(build_mol(symbols[:2], coords[:2], basis))
        mf_B = run_scf(build_mol(symbols[2:], coords[2:], basis))
        E_elst = compute_elst(mf_A, mf_B, mol_AB, basis)
        E_exch = compute_exch(mf_A, mf_B, mol_AB, basis)
        E_ind = compute_ind(mf_A, mf_B, mol_AB, basis)
        E_disp = compute_disp(mf_A, mf_B, mol_AB, basis)
        scan_components.append([E_elst, E_exch, E_ind, E_disp])
    np.save('results/distance_scan.npy', np.array(scan_components))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    components = np.load('results/components.npy')
    ax1.bar(['E_elst', 'E_exch', 'E_ind', 'E_disp'], components)
    ax1.set_title('Reference Geometry')
    scan_components = np.load('results/distance_scan.npy')
    ax2.plot(scan_distances, scan_components[:, 0], label='E_elst')
    ax2.plot(scan_distances, scan_components[:, 1], label='E_exch')
    ax2.plot(scan_distances, scan_components[:, 2], label='E_ind')
    ax2.plot(scan_distances, scan_components[:, 3], label='E_disp')
    ax2.set_xlabel('O–O Distance (Å)')
    ax2.set_ylabel('Energy (kcal/mol)')
    ax2.legend()
    fig.tight_layout()
    plt.savefig('results/decomposition_plot.png')
    plt.close()

if __name__ == '__main__':
    main()