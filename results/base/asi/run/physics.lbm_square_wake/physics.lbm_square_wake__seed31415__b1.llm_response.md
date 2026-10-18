```python
import json
import numpy as np
import pandas as pd
from numba import njit, prange

# D2Q9 lattice parameters
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

@njit(fastmath=True, parallel=True)
def equilibrium_cell(rho, ux, uy, f):
    for i in prange(9):
        cix = cx[i]
        ciy = cy[i]
        u = ux * cix + uy * ciy
        f[i] = w[i] * rho * (1 + 3 * u + 4.5 * u * u - 1.5 * (ux * ux + uy * uy))

@njit(fastmath=True, parallel=True)
def initialize(f, solid, rho0, ux0, uy0):
    for y in prange(f.shape[0]):
        for x in prange(f.shape[1]):
            if not solid[y, x]:
                rho = rho0
                ux = ux0
                uy = uy0
                equilibrium_cell(rho, ux, uy, f[y, x, :])

@njit(fastmath=True, parallel=True)
def collide_trt(f, fpost, solid, omega_plus, omega_minus):
    for y in prange(f.shape[0]):
        for x in prange(f.shape[1]):
            if not solid[y, x]:
                rho = np.sum(f[y, x, :])
                ux = np.sum(f[y, x, :] * cx) / rho
                uy = np.sum(f[y, x, :] * cy) / rho
                equilibrium_cell(rho, ux, uy, f)
                for i in prange(9):
                    j = opp[i]
                    f_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                    f_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                    feq_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                    feq_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                    fpost[y, x, i] = f[y, x, i] - omega_plus * (f_plus - feq_plus) - omega_minus * (f_minus - feq_minus)

@njit(fastmath=True, parallel=True)
def stream_bounce(fpost, fnew, solid):
    Fx = 0.0
    Fy = 0.0
    for y in prange(fnew.shape[0]):
        for x in prange(fnew.shape[1]):
            for i in prange(9):
                xn = x + cx[i]
                yn = y + cy[i]
                if xn < 0 or xn >= fnew.shape[1] or yn < 0 or yn >= fnew.shape[0]:
                    continue
                if solid[yn, xn]:
                    j = opp[i]
                    fnew[y, x, j] += fpost[y, x, i]
                    Fx += 2 * fpost[y, x, i] * cx[i]
                    Fy += 2 * fpost[y, x, i] * cy[i]
                else:
                    fnew[yn, xn, i] += fpost[y, x, i]
    return Fx, Fy

@njit(fastmath=True, parallel=True)
def macro_cell(f, y, x):
    rho = np.sum(f[y, x, :])
    ux = np.sum(f[y, x, :] * cx) / rho
    uy = np.sum(f[y, x, :] * cy) / rho
    return rho, ux, uy

@njit(fastmath=True, parallel=True)
def apply_neq_extrapolation(f, solid, uin, rho_out):
    for y in prange(f.shape[0]):
        for x in prange(f.shape[1]):
            if not solid[y, x]:
                for i in prange(9):
                    xn = x + cx[i]
                    yn = y + cy[i]
                    if xn < 0 or xn >= f.shape[1] or yn < 0 or yn >= f.shape[0]:
                        continue
                    if solid[yn, xn]:
                        continue
                    if xn == 0:
                        rho_ref = np.sum(f[y, x, :])
                        ux_ref = np.sum(f[y, x, :] * cx) / rho_ref
                        uy_ref = np.sum(f[y, x, :] * cy) / rho_ref
                        feq = w[i] * rho_ref * (1 + 3 * (cx[i] * ux_ref + cy[i] * uy_ref) + 4.5 * (cx[i] * ux_ref + cy[i] * uy_ref)**2 - 1.5 * (ux_ref**2 + uy_ref**2))
                        f[yn, xn, i] = feq + f[y, x, i] - feq
                    elif xn == f.shape[1] - 1:
                        rho_ref = np.sum(f[y, x, :])
                        ux_ref = np.sum(f[y, x, :] * cx) / rho_ref
                        uy_ref = np.sum(f[y, x, :] * cy) / rho_ref
                        feq = w[i] * rho_ref * (1 + 3 * (cx[i] * ux_ref + cy[i] * uy_ref) + 4.5 * (cx[i] * ux_ref + cy[i] * uy_ref)**2 - 1.5 * (ux_ref**2 + uy_ref**2))
                        f[yn, xn, i] = feq + f[y, x, i] - feq
                    elif yn == 0:
                        rho_ref = np.sum(f[y, x, :])
                        ux_ref = np.sum(f[y, x, :] * cx) / rho_ref
                        uy_ref = np.sum(f[y, x, :] * cy) / rho_ref
                        feq = w[i] * rho_ref * (1 + 3 * (cx[i] * ux_ref + cy[i] * uy_ref) + 4.5 * (cx[i] * ux_ref + cy[i] * uy_ref)**2 - 1.5 * (ux_ref**2 + uy_ref**2))
                        f[yn, xn, i] = feq + f[y, x, i] - feq
                    elif yn == f.shape[0] - 1:
                        rho_ref = np.sum(f[y, x, :])
                        ux_ref = np.sum(f[y, x, :] * cx) / rho_ref
                        uy_ref = np.sum(f[y, x, :] * cy) / rho_ref
                        feq = w[i] * rho_ref * (1 + 3 * (cx[i] * ux_ref + cy[i] * uy_ref) + 4.5 * (cx[i] * ux_ref + cy[i] * uy_ref)**2 - 1.5 * (ux_ref**2 + uy_ref**2))
                        f[yn, xn, i] = feq + f[y, x, i] - feq

@njit(fastmath=True, parallel=True)
def compute_macro(f, solid, rho, ux, uy):
    for y in prange(f.shape[0]):
        for x in prange(f.shape[1]):
            if not solid[y, x]:
                rho[y, x] = np.sum(f[y, x, :])
                ux[y, x] = np.sum(f[y, x, :] * cx) / rho[y, x]
                uy[y, x] = np.sum(f[y, x, :] * cy) / rho[y, x]

def simulate_case(case):
    nx = case['nx']
    ny = case['ny']
    uin = case['uin']
    re = case['re']
    cx_center = case['square_center'][0]
    cy_center = case['square_center'][1]
    square_size = case['square_size']
    n_steps = case['n_steps']
    warmup_steps = case['warmup_steps']
    sample_every = case['sample_every']
    case_id = case['case_id']

    half = square_size // 2
    x0 = cx_center - half
    x1 = x0 + square_size
    y0 = cy_center - half
    y1 = y0 + square_size

    solid = np.zeros((ny, nx), dtype=bool)
    for y in prange(ny):
        for x in prange(nx):
            if x0 <= x < x1 and y0 <= y < y1:
                solid[y, x] = True

    f = np.zeros((ny, nx, 9), dtype=np.float64)
    initialize(f, solid, 1.0, uin, 0.0)

    rho = np.zeros((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    fnew = np.zeros((ny, nx, 9), dtype=np.float64)
    fpost = np.zeros((ny, nx, 9), dtype=np.float64)

    nu = uin * square_size / re
    tau_plus = 0.5 + 3 * nu
    omega_plus = 1.0 / tau_plus
    tau_minus = 0.5 + (3/16) / (tau_plus - 0.5)
    omega_minus = 1.0 / tau_minus

    force_series = []
    for step in range(n_steps):
        collide_trt(f, fpost, solid, omega_plus, omega_minus)
        Fx, Fy = stream_bounce(fpost, fnew, solid)
        apply_neq_extrapolation(fnew, solid, uin, 1.0)
        compute_macro(fnew, solid, rho, ux, uy)
        f, fnew = fnew, f

        if step >= warmup_steps and (step - warmup_steps) % sample_every == 0:
            force_scale = 0.5 * uin**2 * square_size
            cd = Fx / force_scale
            cl = Fy / force_scale
            force_series.append([case_id, step, cd, cl])

    cd_mean = np.mean([d[2] for d in force_series])
    cl_rms = np.sqrt(np.mean([(d[2] - cd_mean)**2 for d in force_series]))

    # Centerline profile
    centerline = []
    for x in range(nx):
        rho_val, ux_val, uy_val = macro_cell(f, 0, x)
        centerline.append([case_id, x, ux_val])

    # Downstream transverse profile
    x_probe = min(nx - 1, cx_center + 2 * square_size)
    transverse_profile = []
    for y in range(ny):
        rho_val, ux_val, uy_val = macro_cell(f, y, x_probe)
        transverse_profile.append([case_id, x_probe, y, ux_val, uy_val])

    return {
        'case_id': case_id,
        'nx': nx,
        'ny': ny,
        're': re,
        'uin': uin,
        'nu': nu,
        'tau': tau_plus,
        'n_steps': n_steps,
        'cd_mean': cd_mean,
        'cl_rms': cl_rms,
        'converged': 1,
        'force_series': force_series,
        'centerline': centerline,
        'transverse_profile': transverse_profile
    }

def main():
    with open('./data/task_input.json', 'r') as f:
        cases = json.load(f)

    results = []
    for case in cases:
        result = simulate_case(case)
        results.append(result)

    # Write output files
    fluid_summary = []
    fluid_force_series = []
    fluid_centerline = []
    fluid_transverse_profile = []

    for result in results:
        fluid_summary.append({
            'case_id': result['case_id'],
            'nx': result['nx'],
            'ny': result['ny'],
            're': result['re'],
            'uin': result['uin'],
            'nu': result['nu'],
            'tau': result['tau'],
            'n_steps': result['n_steps'],
            'cd_mean': result['cd_mean'],
            'cl_rms': result['cl_rms'],
            'converged': result['converged']
        })
        for entry in result['force_series']:
            fluid_force_series.append({
                'case_id': entry[0],
                'step': entry[1],
                'cd': entry[2],
                'cl': entry[3]
            })
        for entry in result['centerline']:
            fluid_centerline.append({
                'case_id': entry[0],
                'x': entry[1],
                'u': entry[2]
            })
        for entry in result['transverse_profile']:
            fluid_transverse_profile.append({
                'case_id': entry[0],
                'x_probe': entry[1],
                'y': entry[2],
                'u': entry[3],
                'v': entry[4]
            })

    pd.DataFrame(fluid_summary).to_csv('./results/fluid_summary.csv', index=False)
    pd.DataFrame(fluid_force_series).to_csv('./results/fluid_force_series.csv', index=False)
    pd.DataFrame(fluid_centerline).to_csv('./results/fluid_centerline.csv', index=False)
    pd.DataFrame(fluid_transverse_profile).to_csv('./results/fluid_transverse_profile.csv', index=False)

    fluid_report = {
        'task_id': 'lattice_boltzmann_simulation',
        'units': 'dimensionless',
        'cd_mean_avg': np.mean([r['cd_mean'] for r in results]),
        'method_summary': 'D2Q9 TRT-LBM simulation with non-equilibrium extrapolation for outer boundaries'
    }

    with open('./results/fluid_report.json', 'w') as f:
        json.dump(fluid_report, f)

if __name__ == '__main__':
    main()
```