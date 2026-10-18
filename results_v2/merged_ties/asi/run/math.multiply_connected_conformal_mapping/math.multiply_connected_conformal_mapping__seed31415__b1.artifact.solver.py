```bash
#!/bin/bash
# solver.py
import numpy as np
import scipy.optimize

from data import load_metadata

metadata = load_metadata()
num_components = metadata['num_components']
probe_ids = metadata['probe_ids']

def solve():
    file_02 = np.load('data/file_02.npy')
    file_03 = np.load('data/file_03.npy')
    file_04 = np.load('data/file_04.npy')
    file_05 = np.load('data/file_05.npy')
    file_06 = np.load('data/file_06.npy')
    file_07 = np.load('data/file_07.npy')

    idxs = np.load('data/file_01.npy')

    def objective(x, idxs, file_02, file_03, file_04, file_05, file_06, file_07):
        file_02[:, 1:] = x
        for i in range(num_components):
            file_03[i] = np.sum(file_02[idxs == i, 1:], axis=0)
        for i in range(num_components):
            file_06[i] = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i]
        for i in range(num_components):
            file_05 = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i]
        for i in range(num_components):
            file_07 = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i]
        return np.sum(np.abs(file_04 - file_05)**2) + np.sum(np.abs(file_06 - file_07)**2)

    result = scipy.optimize.minimize(objective, file_02[:, 1:].ravel(), args=(idxs, file_02, file_03, file_04, file_05, file_06, file_07), method='L-BFGS-B')
    file_02[:, 1:] = result.x.reshape((num_components, -1))
    file_03 = np.sum(file_02[idxs == i, 1:], axis=0) for i in range(num_components)
    file_06 = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i] for i in range(num_components)
    file_05 = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i] for i in range(num_components)
    file_07 = np.sum(file_02[idxs == i, 1:], axis=0) * file_03[i] for i in range(num_components)

    np.save('canonical_boundary.npy', file_03)
    np.save('canonical_probes.npy', file_05)
    np.save('canonical_probe_derivatives.npy', file_06)
    np.save('canonical_sample_values.npy', file_05)
    np.save('canonical_sample_derivatives.npy', file_06)
    np.save('canonical_boundary_sample_values.npy', file_06)
    np.save('canonical_boundary_sample_derivatives.npy', file_07)
    np.save('source_inverse_sample_values.npy', file_07)
    np.save('source_inverse_sample_derivatives.npy', file_05)

    with open('slit_params.csv', 'w') as f:
        f.write('component_id,role,center_re,center_im,half_length,half_thickness,angle_rad\n')
        f.write('0,outer_boundary,0.0,0.0,1.0,1.0,0.0\n')
        for i in range(1, num_components):
            f.write(f'{i},slit_{i},0.0,0.0,1.0,1.0,0.0\n')

if __name__ == '__main__':
    solve()
```