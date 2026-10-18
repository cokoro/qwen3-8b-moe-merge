import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, lib
from pyscf import ao2mo
from scipy.linalg import block_diag
import os
import json

def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    atoms = []
    for line in lines:
        if line.strip() and not line.startswith('#'):
            parts = line.strip().split()
            atoms.append((parts[0], tuple(map(float, parts[1:]))))
    return atoms

def write_xyz(file_path, atoms, comment=''):
    with open(file_path, 'w') as f:
        f.write(comment + '\n')
        f.write(str(len(atoms)) + '\n')
        for atom in atoms:
            f.write(f"{atom[0]} {' '.join(map(str, atom[1]))}\n")

def build_ghost_molecule(mol, other_mol, ghost_label):
    ghost_atoms = [(ghost_label, pos) for (sym, pos) in other_mol]
    return gto.M(atom=mol.atom + ghost_atoms, basis=mol.basis, unit=mol.unit)

def compute_sapt0_components(mol_A, mol_B, basis, scan_distance):
    # Step 1: Monomer SCFs in DCBS
    mol_A_ghost = build_ghost_molecule(mol_A, mol_B, 'ghost-B')
    mol_B_ghost = build_ghost_molecule(mol_B, mol_A, 'ghost-A')
    
    mf_A = scf.RHF(mol_A_ghost).run()
    mf_B = scf.RHF(mol_B_ghost).run()
    
    C_A_occ = mf_A.mo_coeff[:, :mf_A.mol.nelectron//2]
    C_A_vir = mf_A.mo_coeff[:, mf_A.mol.nelectron//2:]
    C_B_occ = mf_B.mo_coeff[:, :mf_B.mol.nelectron//2]
    C_B_vir = mf_B.mo_coeff[:, mf_B.mol.nelectron//2:]
    
    eps_A = mf_A.mo_energy[:mf_A.mol.nelectron//2]
    eps_B = mf_B.mo_energy[:mf_B.mol.nelectron//2]
    
    # Step 2: E_elst
    S_AO = mol_A_ghost.get_ovlp()
    V_nuc_AB = mol_A_ghost.intor('int1e_nuc') + mol_B_ghost.intor('int1e_nuc')
    V_nuc_AB = V_nuc_AB - np.diag(V_nuc_AB.diagonal()[:len(mol_A.atom)] + V_nuc_AB.diagonal()[len(mol_A.atom):])
    D_A = 2 * np.dot(C_A_occ, C_A_occ.T)
    D_B = 2 * np.dot(C_B_occ, C_B_occ.T)
    
    V_nuc_AB = np.dot(D_A, V_nuc_AB) + np.dot(D_B, V_nuc_AB)
    V_nuc_AB = np.diag(V_nuc_AB)
    
    J = mol_A_ghost.intor('int2e')
    J = ao2mo.restore(1, J, mol_A_ghost.nao)
    J = np.einsum('ijab,klcd->ijkl', J, J)
    J = np.einsum('ijkl,abij->abkl', J, D_A)
    J = np.einsum('ijkl,abij->abkl', J, D_B)
    J = np.diag(J)
    
    E_elst = np.sum(V_nuc_AB) + np.sum(J)
    
    # Step 3: E_exch
    S_AB = np.dot(C_A_occ.T, np.dot(S_AO, C_B_occ))
    S_AB = S_AB + S_AB.T
    S_AB = S_AB / 2
    
    E_exch = -np.trace(np.dot(S_AB, (np.dot(D_A, mol_A_ghost.intor('int1e_nuc')) + np.dot(D_B, mol_B_ghost.intor('int1e_nuc')))))
    
    # Step 4: E_ind
    def solve_cphf(mol, other_mol, D_other):
        nocc = mol.nocc
        nvir = mol.nvir
        F = mol.get_fock()
        V = mol.intor('int1e_rhf')
        V = np.dot(V, D_other)
        V = V + np.dot(V.T, D_other)
        V = V - np.diag(V.diagonal())
        K = mol.intor('int2e')
        K = ao2mo.restore(1, K, mol.nao)
        K = np.einsum('ijab,klcd->ijkl', K, K)
        K = np.einsum('ijkl,abij->abkl', K, D_other)
        K = np.einsum('ijkl,abij->abkl', K, D_other)
        K = np.diag(K)
        A = np.diag(eps_A - eps_B)
        B = np.diag(eps_A - eps_B)
        C = np.dot(D_other, V)
        D = np.dot(D_other, K)
        E = np.dot(D_other, V)
        F = np.dot(D_other, K)
        H = np.block([[A, C], [E, B]])
        G = np.block([[C, D], [E, F]])
        t = np.linalg.solve(H, G)
        t = t[:nocc, nocc:]
        E_ind = -2 * np.sum((eps_A - eps_B) * t**2)
        return E_ind
    
    E_ind = solve_cphf(mol_A, mol_B, D_B) + solve_cphf(mol_B, mol_A, D_A)
    
    # Step 5: E_disp
    def compute_disp(mol_A, mol_B):
        nocc = mol_A.nocc
        nvir = mol_A.nvir
        nocc_B = mol_B.nocc
        nvir_B = mol_B.nvir
        ints = mol_A.intor('int2e')
        ints = ao2mo.restore(1, ints, mol_A.nao)
        ints = np.einsum('ijab,klcd->ijkl', ints, ints)
        ints = np.einsum('ijkl,abij->abkl', ints, D_A)
        ints = np.einsum('ijkl,abij->abkl', ints, D_B)
        ints = np.diag(ints)
        E_disp = -4 * np.sum(ints / (eps_A + eps_B - eps_A - eps_B))
        return E_disp
    
    E_disp = compute_disp(mol_A, mol_B)
    
    return E_elst, E_exch, E_ind, E_disp

def main():
    config = json.load(open('data/config.json'))
    basis = config['basis']
    scan_distances = config['scan_distances']
    
    dimer_xyz = read_xyz('data/dimer.xyz')
    scan_xyz = read_xyz('data/scan_geometries.xyz')
    
    # Reference geometry
    mol_A = gto.M(atom=dimer_xyz[:3], basis=basis, unit='Angstrom')
    mol_B = gto.M(atom=dimer_xyz[3:], basis=basis, unit='Angstrom')
    
    E_elst_ref, E_exch_ref, E_ind_ref, E_disp_ref = compute_sapt0_components(mol_A, mol_B, basis, scan_distances[0])
    components_ref = np.array([E_elst_ref, E_exch_ref, E_ind_ref, E_disp_ref]) / 23.674  # kcal/mol
    
    # Scan geometries
    results = []
    for i, dist in enumerate(scan_distances):
        mol_A_scan = gto.M(atom=scan_xyz[i][:3], basis=basis, unit='Angstrom')
        mol_B_scan = gto.M(atom=scan_xyz[i][3:], basis=basis, unit='Angstrom')
        E_elst, E_exch, E_ind, E_disp = compute_sapt0_components(mol_A_scan, mol_B_scan, basis, dist)
        components = np.array([E_elst, E_exch, E_ind, E_disp]) / 23.674
        results.append(components)
    
    np.save('results/components.npy', components_ref)
    np.save('results/distance_scan.npy', np.array(results))
    np.save('results/total_interaction.npy', np.sum(components_ref))
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    components = np.load('results/components.npy')
    ax1.bar(['E_elst', 'E_exch', 'E_ind', 'E_disp'], components)
    ax1.set_title('Reference Geometry')
    ax1.set_ylabel('Energy (kcal/mol)')
    
    scan_results = np.load('results/distance_scan.npy')
    dists = np.array(scan_distances)
    for i, comp in enumerate(['E_elst', 'E_exch', 'E_ind', 'E_disp']):
        ax2.plot(dists, scan_results[:, i], label=comp)
    ax2.set_xlabel('O–O Distance (Å)')
    ax2.set_ylabel('Energy (kcal/mol)')
    ax2.legend()
    fig.tight_layout()
    plt.savefig('results/decomposition_plot.png')

if __name__ == '__main__':
    main()