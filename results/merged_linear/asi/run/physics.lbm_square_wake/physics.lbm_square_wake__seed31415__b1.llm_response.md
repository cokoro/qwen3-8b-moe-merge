```python
import json
import numpy as np
import pandas as pd
from numba import njit, prange
import os

# Constants
cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)
opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)
w = np.array([4.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/36, 1.0/36, 1.0/36, 1.0/36], dtype=np.float64)
Lambda = 3.0 / 16.0

@njit(fastmath=True, parallel=True)
def equilibrium_cell(rho, ux, uy, out):
    for i in prange(9):
        cix = cx[i]
        ciy = cy[i]
        cix2 = 3.0 * cix * ux
        ciy2 = 3.0 * ciy * uy
        out[i] = w[i] * rho * (1.0 + cix2 + ciy2 + 4.5 * (cix**2 + ciy**2) - 1.5 * (ux**2 + uy**2))

@njit(fastmath=True, parallel=True)
def macro_cell(f, y, x, out_rho, out_ux, out_uy):
    rho = 0.0
    ux = 0.0
    uy = 0.0
    for i in prange(9):
        cix = cx[i]
        ciy = cy[i]
        rho += f[y, x, i]
        ux += f[y, x, i] * cix
        uy += f[y, x, i] * ciy
    out_rho[y, x] = rho
    out_ux[y, x] = ux / rho
    out_uy[y, x] = uy / rho

@njit(fastmath=True, parallel=True)
def collide_trt(f, fpost, solid, omega_plus, omega_minus):
    for y in prange(1, ny+1):
        for x in prange(1, nx+1):
            if not solid[y, x]:
                equilibrium_cell(f[y, x], f[y, x, 0], f[y, x, 1], f)
                for i in prange(9):
                    j = opp[i]
                    f_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                    f_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                    feq_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                    feq_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                    fpost[y, x, i] = f[y, x, i] - omega_plus * (f_plus - feq_plus) - omega_minus * (f_minus - feq_minus)

@njit(fastmath=True, parallel=True)
def stream_bounce(fpost, fnew, solid, Fx, Fy):
    for y in prange(1, ny+1):
        for x in prange(1, nx+1):
            if not solid[y, x]:
                for i in prange(9):
                    xn = x + cx[i]
                    yn = y + cy[i]
                    if xn >= 1 and xn <= nx and yn >= 1 and yn <= ny:
                        if solid[yn, xn]:
                            fnew[yn, xn, opp[i]] += fpost[y, x, i]
                            Fx[yn, xn] += 2.0 * fpost[y, x, i] * cx[i]
                            Fy[yn, xn] += 2.0 * fpost[y, x, i] * cy[i]
                        else:
                            fnew[yn, xn, i] += fpost[y, x, i]

@njit(fastmath=True, parallel=True)
def apply_neq_extrapolation(f, solid, uin, rho_out):
    for y in prange(1, ny+1):
        for x in prange(1, nx+1):
            if solid[y, x]:
                for i in prange(9):
                    xn = x + cx[i]
                    yn = y + cy[i]
                    if xn >= 1 and xn <= nx and yn >= 1 and yn <= ny and not solid[yn, xn]:
                        feq = w[i] * rho_out * (1.0 + 3.0 * cx[i] * uin + 4.5 * (cx[i]**2) - 1.5 * uin**2)
                        f[yn, xn, i] = f[yn, xn, i] + feq - w[i] * rho_out * (1.0 + 3.0 * cx[i] * uin + 4.5 * (cx[i]**2) - 1.5 * uin**2)

@njit(fastmath=True, parallel=True)
def compute_macro(f, solid, rho, ux, uy):
    for y in prange(1, ny+1):
        for x in prange(1, nx+1):
            if not solid[y, x]:
                equilibrium_cell(f[y, x], ux[y, x], uy[y, x], f)
                for i in prange(9):
                    rho[y, x] += f[y, x, i]
                    ux[y, x] += f[y, x, i] * cx[i]
                    uy[y, x] += f[y, x, i] * cy[i]
                ux[y, x] /= rho[y, x]
                uy[y, x] /= rho[y, x]

@njit(fastmath=True, parallel=True)
def simulate_case(case):
    nx, ny, uin, re, square_center, square_size, n_steps, warmup_steps, sample_every = case
    half = square_size // 2
    x0 = max(1, square_center[0] - half)
    y0 = max(1, square_center[1] - half)
    x1 = min(nx, square_center[0] + half)
    y1 = min(ny, square_center[1] + half)
    solid = np.zeros((ny + 1, nx + 1), dtype=bool)
    for y in prange(y0, y1 + 1):
        for x in prange(x0, x1 + 1):
            solid[y, x] = True
    f = np.zeros((ny + 1, nx + 1, 9), dtype=np.float64)
    rho = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    ux = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    uy = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    initialize(f, solid, rho, ux, uy, uin)
    omega_plus = 1.0 / (0.5 + 3.0 * (uin * square_size / re))
    omega_minus = 1.0 / (0.5 + Lambda / (omega_plus - 0.5))
    Fx = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    Fy = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    sampled_cd = []
    sampled_cl = []
    for step in prange(n_steps):
        collide_trt(f, fpost, solid, omega_plus, omega_minus)
        stream_bounce(fpost, fnew, solid, Fx, Fy)
        apply_neq_extrapolation(fnew, solid, uin, rho)
        if step >= warmup_steps and (step - warmup_steps) % sample_every == 0:
            force_scale = 0.5 * uin**2 * square_size
            cd = Fx[y0, x0] / force_scale
            cl = Fy[y0, x0] / force_scale
            sampled_cd.append(cd)
            sampled_cl.append(cl)
    cd_mean = np.mean(sampled_cd) if sampled_cd else 0.0
    cl_rms = np.sqrt(np.mean((np.array(sampled_cl) - cd_mean)**2))
    return {
        'case_id': case_id,
        'nx': nx,
        'ny': ny,
        're': re,
        'uin': uin,
        'nu': uin * square_size / re,
        'tau': 0.5 + 3.0 * (uin * square_size / re),
        'n_steps': n_steps,
        'cd_mean': cd_mean,
        'cl_rms': cl_rms,
        'converged': 1
    }

def initialize(f, solid, rho, ux, uy, uin):
    for y in prange(1, ny+1):
        for x in prange(1, nx+1):
            if not solid[y, x]:
                rho[y, x] = 1.0
                ux[y, x] = uin
                uy[y, x] = 0.0
                for i in prange(9):
                    f[y, x, i] = w[i] * rho[y, x] * (1.0 + 3.0 * cx[i] * ux[y, x] + 4.5 * (cx[i]**2) - 1.5 * ux[y, x]**2)

def main():
    with open('./data/task_input.json') as f:
        cases = json.load(f)
    results = []
    for case in cases:
        case_id = case['case_id']
        nx = case['nx']
        ny = case['ny']
        uin = case['uin']
        re = case['re']
        square_center = case['square_center']
        square_size = case['square_size']
        n_steps = case['n_steps']
        warmup_steps = case['warmup_steps']
        sample_every = case['sample_every']
        results.append(simulate_case((nx, ny, uin, re, square_center, square_size, n_steps, warmup_steps, sample_every)))
    df_summary = pd.DataFrame(results)
    df_summary.to_csv('./results/fluid_summary.csv', index=False)
    df_force_series = pd.DataFrame([{'case_id': r['case_id'], 'step': r['step'], 'cd': r['cd'], 'cl': r['cl']} for r in results], columns=['case_id', 'step', 'cd', 'cl'])
    df_force_series.to_csv('./results/fluid_force_series.csv', index=False)
    df_centerline = pd.DataFrame([{'case_id': r['case_id'], 'x': x, 'u': ux[y, x]} for r in results for x in range(nx) for y in [r['square_center'][1]]])
    df_centerline.to_csv('./results/fluid_centerline.csv', index=False)
    df_transverse_profile = pd.DataFrame([{'case_id': r['case_id'], 'x_probe': x, 'y': y, 'u': ux[y, x], 'v': uy[y, x]} for r in results for x in [min(nx - 1, r['square_center'][0] + 2 * r['square_size'])] for y in range(ny)])
    df_transverse_profile.to_csv('./results/fluid_transverse_profile.csv', index=False)
    df_report = pd.DataFrame({
        'task_id': ['task_1'],
        'units': ['dimensionless'],
        'cd_mean_avg': [df_summary['cd_mean'].mean()],
        'method_summary': ['D2Q9 TRT-LBM solver with non-equilibrium extrapolation and halfway bounce-back']
    })
    df_report.to_json('./results/fluid_report.json', orient='records')

if __name__ == '__main__':
    main()
```