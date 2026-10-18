```python
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo, lib

def load_geometries(file_path):
    geometries = []
    with open(file_path) as f:
        lines = f.read().splitlines()
    frame = []
    for i, line in enumerate(lines):
        if i == 0:
            nframes = int(line)
        elif i % (nframes + 1) == 0:
            geometries.append(frame)
            frame = []
        else:
            frame.append(line)
    geometries.append(frame)
    geometries = [np.array([list(map(float, line.split())) for line in geom]) for geom in geometries]
    return geometries[1:]

def read_config(config_path):
    import json
    with open(config_path) as f:
        config = json.load(f)
    return config

def build_molecule(mol_str, basis, unit='Angstrom'):
    mol = gto.M(
        atom=mol_str,
        basis=basis,
        unit=unit,
        verbose=0
    )
    return mol

def compute_scf(mol, charge=0, spin=0):
    mf = scf.RHF(mol).run()
    return mf

def compute_overlap(mol):
    return mol.intor('int1e_ovlp')

def compute_nuclear_attraction(mol):
    return mol.intor('int1e_nuc')

def compute_two_electron_integrals(mol):
    return ao2mo.kernel(mol, mol._eri)

def compute_one_electron_density(mf, occ_indices):
    mo_occ = mf.mo_coeff[:, occ_indices]
    return 2 * mo_occ @ mo_occ.T

def compute_electron_electron_repulsion(mf, occ_indices):
    mo_occ = mf.mo_coeff[:, occ_indices]
    return ao2mo.kernel(mf, mf._eri, mo_occ)

def compute_e_elst(mf_A, mf_B, basis, charge=0, spin=0):
    S_AB = compute_overlap(basis)
    D_A = compute_one_electron_density(mf_A, mf_A.mo_occ)
    D_B = compute_one_electron_density(mf_B, mf_B.mo_occ)
    V_nuc_AB = compute_nuclear_attraction(basis)
    V_electron_electron_AB = compute_electron_electron_repulsion(mf_A, mf_A.mo_occ)
    E_elst = np.trace(D_A @ V_nuc_AB) + np.trace(D_B @ V_nuc_AB)
    E_elst += np.trace(D_A @ V_electron_electron_AB)
    return E_elst

def compute_e_exch(mf_A, mf_B, basis):
    S_AB = compute_overlap(basis)
    D_A = compute_one_electron_density(mf_A, mf_A.mo_occ)
    D_B = compute_one_electron_density(mf_B, mf_B.mo_occ)
    E_exch = -np.trace(S_AB @ (D_A @ compute_nuclear_attraction(basis) + D_B @ compute_nuclear_attraction(basis)))
    return E_exch

def compute_e_ind(mf_A, mf_B, basis):
    nocc_A = mf_A.mo_occ.shape[0]
    nocc_B = mf_B.mo_occ.shape[0]
    nmo_A = mf_A.mo_coeff.shape[1]
    nmo_B = mf_B.mo_coeff.shape[1]
    V_B = mf_B.get_fock()
    V_A = mf_A.get_fock()
    C_A = mf_A.mo_coeff
    C_B = mf_B.mo_coeff
    S_AB = compute_overlap(basis)
    S_AB = S_AB.reshape((nmo_A, nmo_B))
    t_A = np.zeros((nocc_A, nmo_A))
    t_B = np.zeros((nocc_B, nmo_B))
    for i in range(nocc_A):
        for a in range(nmo_A):
            if a < nocc_A:
                continue
            rhs = np.zeros(nmo_A)
            rhs[a] = 1.0
            t_A[i, a] = solve_cphf(rhs, C_A, V_A, S_AB, C_B, V_B)
    for j in range(nocc_B):
        for b in range(nmo_B):
            if b < nocc_B:
                continue
            rhs = np.zeros(nmo_B)
            rhs[b] = 1.0
            t_B[j, b] = solve_cphf(rhs, C_B, V_B, S_AB.T, C_A, V_A)
    E_ind = -2.0 * np.sum((np.diag(V_A) - np.diag(V_B)) * (t_A ** 2))
    E_ind += -2.0 * np.sum((np.diag(V_B) - np.diag(V_A)) * (t_B ** 2))
    return E_ind

def solve_cphf(rhs, C, V, S_AB, C_B, V_B):
    nocc = C.shape[0]
    nmo = C.shape[1]
    S_AB = S_AB.reshape((nmo, nmo))
    F = np.diag(V) + C.T @ C
    F = F.reshape((nmo, nmo))
    rhs = rhs.reshape((nmo, 1))
    t = np.linalg.solve(F, rhs)
    return t[0]

def compute_e_disp(mf_A, mf_B, basis):
    nocc_A = mf_A.mo_occ.shape[0]
    nocc_B = mf_B.mo_occ.shape[0]
    nmo_A = mf_A.mo_coeff.shape[1]
    nmo_B = mf_B.mo_coeff.shape[1]
    V_AB = mf_A._eri
    V_AB = ao2mo.restore(1, V_AB, nmo_A)
    V_AB = V_AB.reshape((nmo_A, nmo_A, nmo_B, nmo_B))
    E_disp = 0.0
    for i in range(nocc_A):
        for j in range(nocc_B):
            for a in range(nmo_A):
                for b in range(nmo_B):
                    if a < nocc_A or b < nocc_B:
                        continue
                    E_disp += np.abs(V_AB[i, a, j, b]) ** 2 / (np.diag(V_A)[a] + np.diag(V_B)[b] - np.diag(V_A)[i] - np.diag(V_B)[j])
    return E_disp

def compute_components(mol_A, mol_B, basis, charge=0, spin=0):
    mf_A = compute_scf(mol_A, charge, spin)
    mf_B = compute_scf(mol_B, charge, spin)
    E_elst = compute_e_elst(mf_A, mf_B, basis, charge, spin)
    E_exch = compute_e_exch(mf_A, mf_B, basis)
    E_ind = compute_e_ind(mf_A, mf_B, basis)
    E_disp = compute_e_disp(mf_A, mf_B, basis)
    return E_elst, E_exch, E_ind, E_disp

def main():
    config = read_config('data/config.json')
    geometries = load_geometries('data/scan_geometries.xyz')
    basis = config['basis']
    charge = 0
    spin = 0
    components = []
    for geometry in geometries:
        mol_str = [('H', (geometry[0, 0], geometry[0, 1], geometry[0, 2])),
                   ('H', (geometry[1, 0], geometry[1, 1], geometry[1, 2])),
                   ('O', (geometry[2, 0], geometry[2, 1], geometry[2, 2])),
                   ('H', (geometry[3, 0], geometry[3, 1], geometry[3, 2])),
                   ('H', (geometry[4, 0], geometry[4, 1], geometry[4, 2])),
                   ('O', (geometry[5, 0], geometry[5, 1], geometry[5, 2]))]
        mol_A = build_molecule(mol_str[:3] + [('ghost-O', (geometry[5, 0], geometry[5, 1], geometry[5, 2]))], basis)
        mol_B = build_molecule(mol_str[3:] + [('ghost-O', (geometry[2, 0], geometry[2, 1], geometry[2, 2]))], basis)
        E_elst, E_exch, E_ind, E_disp = compute_components(mol_A, mol_B, basis, charge, spin)
        components.append([E_elst, E_exch, E_ind, E_disp])
    components = np.array(components)
    np.save('results/distance_scan.npy', components)
    reference_geometry = load_geometries('data/dimer.xyz')[0]
    mol_str = [('H', (reference_geometry[0, 0], reference_geometry[0, 1], reference_geometry[0, 2])),
               ('H', (reference_geometry[1, 0], reference_geometry[1, 1], reference_geometry[1, 2])),
               ('O', (reference_geometry[2, 0], reference_geometry[2, 1], reference_geometry[2, 2])),
               ('H', (reference_geometry[3, 0], reference_geometry[3, 1], reference_geometry[3, 2])),
               ('H', (reference_geometry[4, 0], reference_geometry[4, 1], reference_geometry[4, 2])),
               ('O', (reference_geometry[5, 0], reference_geometry[5, 1], reference_geometry[5, 2]))]
    mol_A = build_molecule(mol_str[:3] + [('ghost-O', (reference_geometry[5, 0], reference_geometry[5, 1], reference_geometry[5, 2]))], basis)
    mol_B = build_molecule(mol_str[3:] + [('ghost-O', (reference_geometry[2, 0], reference_geometry[2, 1], reference_geometry[2, 2]))], basis)
    E_elst, E_exch, E_ind, E_disp = compute_components(mol_A, mol_B, basis, charge, spin)
    components_ref = np.array([E_elst, E_exch, E_ind, E_disp])
    np.save('results/components.npy', components_ref)
    total_interaction = np.sum(components_ref)
    np.save('results/total_interaction.npy', total_interaction)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].bar(['E_elst', 'E_exch', 'E_ind', 'E_disp'], components_ref)
    ax[0].set_xlabel('Component')
    ax[0].set_ylabel('Energy (kcal/mol)')
    ax[0].set_title('Reference Geometry')
    ax[1].plot(components[:, 0], label='E_elst')
    ax[1].plot(components[:, 1], label='E_exch')
    ax[1].plot(components[:, 2], label='E_ind')
    ax[1].plot(components[:, 3], label='E_disp')
    ax[1].set_xlabel('Distance Index')
    ax[1].set_ylabel('Energy (kcal/mol)')
    ax[1].set_title('Distance Scan')
    ax[1].legend()
    plt.tight_layout()
    plt.savefig('results/decomposition_plot.png')
    plt.close()

if __name__ == '__main__':
    main()
```