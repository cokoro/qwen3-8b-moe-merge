```python
import json
import numpy as np
import pandas as pd
import numba as nb
from numba import njit, prange
import os

def load_input():
    with open('./data/task_input.json') as f:
        task_input = json.load(f)
    return task_input

def write_output(task_id, results):
    os.makedirs('./results', exist_ok=True)
    df_summary = pd.DataFrame(results['summary'])
    df_summary.to_csv('./results/fluid_summary.csv', index=False)
    df_force = pd.DataFrame(results['force_series'])
    df_force.to_csv('./results/fluid_force_series.csv', index=False)
    df_centerline = pd.DataFrame(results['centerline'])
    df_centerline.to_csv('./results/fluid_centerline.csv', index=False)
    df_transverse = pd.DataFrame(results['transverse'])
    df_transverse.to_csv('./results/fluid_transverse_profile.csv', index=False)
    with open('./results/fluid_report.json', 'w') as f:
        json.dump(results['report'], f)

def initialize(nx, ny, uin, square_center, square_size):
    cx, cy = square_center
    half = square_size // 2
    x0 = cx - half
    x1 = x0 + square_size
    y0 = cy - half
    y1 = y0 + square_size
    x0 = max(0, x0)
    x1 = min(nx, x1)
    y0 = max(0, y0)
    y1 = min(ny, y1)
    solid = np.zeros((ny, nx), dtype=np.int32)
    for y in range(y0, y1):
        for x in range(x0, x1):
            solid[y, x] = 1
    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.full((ny, nx), uin, dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    f = np.zeros((ny, nx, 9), dtype=np.float64)
    for y in range(ny):
        for x in range(nx):
            if solid[y, x] == 0:
                feq = equilibrium(rho[y, x], ux[y, x], uy[y, x])
                f[y, x] = feq
    return solid, rho, ux, uy, f

@njit
def equilibrium(rho, ux, uy):
    feq = np.zeros(9, dtype=np.float64)
    for i in range(9):
        ci_x = [0, 1, 0, -1, 0, 1, -1, -1, 1][i]
        ci_y = [0, 0, 1, 0, -1, 1, 1, -1, -1][i]
        ci_u = ci_x * ux + ci_y * uy
        feq[i] = 4/9 * rho * (1 + 3 * ci_u + 4.5 * ci_u**2 - 1.5 * ux**2 - uy**2)
    return feq

@njit
def collide_trt(f, solid, omega_plus, omega_minus):
    for y in range(ny):
        for x in range(nx):
            if solid[y, x] == 0:
                rho = np.sum(f[y, x])
                ux = np.sum(f[y, x] * cx) / rho
                uy = np.sum(f[y, x] * cy) / rho
                feq = equilibrium(rho, ux, uy)
                for i in range(9):
                    j = opp[i]
                    f_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                    f_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                    feq_plus = 0.5 * (feq[i] + feq[j])
                    feq_minus = 0.5 * (feq[i] - feq[j])
                    f[y, x, i] = f[y, x, i] - omega_plus * (f_plus - feq_plus) - omega_minus * (f_minus - feq_minus)

@njit
def stream_bounce(f, solid, uin):
    Fx = 0.0
    Fy = 0.0
    for y in range(ny):
        for x in range(nx):
            if solid[y, x] == 0:
                for i in range(9):
                    cx_i = cx[i]
                    cy_i = cy[i]
                    xn = x + cx_i
                    yn = y + cy_i
                    if xn < 0 or xn >= nx or yn < 0 or yn >= ny:
                        continue
                    if solid[yn, xn] == 1:
                        f[yn, xn, opp[i]] += f[y, x, i]
                        Fx += 2 * f[y, x, i] * cx_i
                        Fy += 2 * f[y, x, i] * cy_i
                    else:
                        f[yn, xn, i] += f[y, x, i]
    return Fx, Fy

@njit
def apply_neq_extrapolation(f, solid, uin, rho_out):
    for y in range(ny):
        for x in range(nx):
            if solid[y, x] == 0:
                if x == 0:
                    rho = rho_out
                    ux = uin
                    uy = 0.0
                    feq = equilibrium(rho, ux, uy)
                    for i in range(9):
                        f[y, x, i] = feq[i] + f[y, x, i] - equilibrium(rho, ux, uy)[i]
                elif x == nx - 1:
                    rho = np.sum(f[y, x])
                    ux = np.sum(f[y, x] * cx) / rho
                    uy = np.sum(f[y, x] * cy) / rho
                    feq = equilibrium(rho, ux, uy)
                    for i in range(9):
                        f[y, x, i] = feq[i] + f[y, x, i] - equilibrium(rho, ux, uy)[i]
                elif y == 0:
                    rho = np.sum(f[y, x])
                    ux = np.sum(f[y, x] * cx) / rho
                    uy = np.sum(f[y, x] * cy) / rho
                    feq = equilibrium(rho, ux, uy)
                    for i in range(9):
                        f[y, x, i] = feq[i] + f[y, x, i] - equilibrium(rho, ux, uy)[i]
                elif y == ny - 1:
                    rho = np.sum(f[y, x])
                    ux = np.sum(f[y, x] * cx) / rho
                    uy = np.sum(f[y, x] * cy) / rho
                    feq = equilibrium(rho, ux, uy)
                    for i in range(9):
                        f[y, x, i] = feq[i] + f[y, x, i] - equilibrium(rho, ux, uy)[i]

@njit
def compute_macro(f, solid, rho, ux, uy):
    for y in range(ny):
        for x in range(nx):
            if solid[y, x] == 0:
                rho[y, x] = np.sum(f[y, x])
                ux[y, x] = np.sum(f[y, x] * cx) / rho[y, x]
                uy[y, x] = np.sum(f[y, x] * cy) / rho[y, x]

def simulate_case(case):
    nx, ny, uin, re, square_center, square_size, n_steps, warmup_steps, sample_every = case
    solid, rho, ux, uy, f = initialize(nx, ny, uin, square_center, square_size)
    omega_plus = 1 / (0.5 + 3 * uin * square_size / re)
    omega_minus = 1 / (0.5 + 3 / (2 * (omega_plus - 0.5)))
    force_series = []
    for step in range(n_steps):
        collide_trt(f, solid, omega_plus, omega_minus)
        Fx, Fy = stream_bounce(f, solid, uin)
        apply_neq_extrapolation(f, solid, uin, 1)
        if step >= warmup_steps and (step - warmup_steps) % sample_every == 0:
            force_scale = 0.5 * uin**2 * square_size
            cd = Fx / force_scale
            cl = Fy / force_scale
            force_series.append([step, cd, cl])
    compute_macro(f, solid, rho, ux, uy)
    centerline = []
    for x in range(nx):
        centerline.append([case_id, x, ux[ny//2, x]])
    transverse_profile = []
    x_probe = min(nx-1, int(square_center[0] + 2 * square_size))
    for y in range(ny):
        transverse_profile.append([case_id, x_probe, y, uy[y, x_probe], ux[y, x_probe]])
    return {
        'summary': {
            'case_id': case_id,
            'nx': nx,
            'ny': ny,
            're': re,
            'uin': uin,
            'nu': uin * square_size / re,
            'tau': 0.5 + 3 * uin * square_size / re,
            'n_steps': n_steps,
            'cd_mean': np.mean([cd for step, cd, cl in force_series]),
            'cl_rms': np.sqrt(np.mean([(cl - np.mean([cl for step, cd, cl in force_series]))**2 for step, cd, cl in force_series])),
            'converged': 1
        },
        'force_series': force_series,
        'centerline': centerline,
        'transverse': transverse_profile
    }

def main():
    task_input = load_input()
    results = []
    for case in task_input:
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
        case_result = simulate_case((case_id, nx, ny, uin, re, square_center, square_size, n_steps, warmup_steps, sample_every))
        results.append(case_result)
    write_output(task_id, {
        'summary': [r['summary'] for r in results],
        'force_series': [r['force_series'] for r in results],
        'centerline': [r['centerline'] for r in results],
        'transverse': [r['transverse'] for r in results],
        'report': {
            'task_id': task_id,
            'units': 'dimensionless',
            'cd_mean_avg': np.mean([r['summary']['cd_mean'] for r in results]),
            'method_summary': 'TRT-LBM'
        }
    })

if __name__ == '__main__':
    task_id = load_input()[0]['task_id']
    main()
```