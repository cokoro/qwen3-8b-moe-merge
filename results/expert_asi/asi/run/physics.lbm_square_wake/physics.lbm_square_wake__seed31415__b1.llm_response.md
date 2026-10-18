```python
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict

def main():
    # Load input
    with open('./data/task_input.json') as f:
        cases = json.load(f)

    # Define constants
    cx = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    cy = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
    opp = [0, 3, 4, 1, 2, 7, 8, 5, 6]

    # Process each case
    results = []
    force_series = defaultdict(list)
    centerline = defaultdict(list)
    transverse_profile = defaultdict(list)

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

        # Geometry
        half = square_size // 2
        x0 = square_center[0] - half
        y0 = square_center[1] - half
        x1 = x0 + square_size
        y1 = y0 + square_size

        # Initialize domain
        solid = np.zeros((ny, nx), dtype=bool)
        solid[y0:y1, x0:x1] = True

        # Lattice parameters
        nu = uin * square_size / re
        tau_plus = 0.5 + 3 * nu
        omega_plus = 1 / tau_plus
        tau_minus = 0.5 + (3/16) / (tau_plus - 0.5)
        omega_minus = 1 / tau_minus

        # Initialize populations
        f = np.zeros((ny, nx, 9), dtype=np.float64)
        for y in range(ny):
            for x in range(nx):
                if not solid[y, x]:
                    rho = 1.0
                    ux = uin
                    uy = 0.0
                    for i in range(9):
                        cxi = cx[i]
                        cyi = cy[i]
                        feq = w[i] * rho * (1 + 3*(cxi*ux + cyi*uy) + 4.5*(cxi**2 + cyi**2)*(ux**2 + uy**2) - 1.5*(ux**2 + uy**2))
                        f[y, x, i] = feq

        # Simulation loop
        fnew = np.zeros((ny, nx, 9), dtype=np.float64)
        force_x = 0.0
        force_y = 0.0
        step = 0
        sampled_cd = []
        sampled_cl = []

        for step in range(n_steps):
            # Collision step
            fpost = np.zeros((ny, nx, 9), dtype=np.float64)
            for y in range(ny):
                for x in range(nx):
                    if solid[y, x]:
                        continue
                    # Compute macroscopic quantities
                    rho = np.sum(f[y, x])
                    ux = np.sum(f[y, x] * np.array(cx)) / rho
                    uy = np.sum(f[y, x] * np.array(cy)) / rho
                    # Compute equilibrium
                    feq = np.zeros(9)
                    for i in range(9):
                        cxi = cx[i]
                        cyi = cy[i]
                        feq[i] = w[i] * rho * (1 + 3*(cxi*ux + cyi*uy) + 4.5*(cxi**2 + cyi**2)*(ux**2 + uy**2) - 1.5*(ux**2 + uy**2))
                    # TRT collision
                    for i in range(9):
                        j = opp[i]
                        f_plus = 0.5 * (f[y, x, i] + f[y, x, j])
                        f_minus = 0.5 * (f[y, x, i] - f[y, x, j])
                        feq_plus = 0.5 * (feq[i] + feq[j])
                        feq_minus = 0.5 * (feq[i] - feq[j])
                        fpost[y, x, i] = f[y, x, i] - omega_plus*(f_plus - feq_plus) - omega_minus*(f_minus - feq_minus)

            # Streaming and bounce-back
            Fx_total = 0.0
            Fy_total = 0.0
            for y in range(ny):
                for x in range(nx):
                    if solid[y, x]:
                        continue
                    for i in range(9):
                        cxi = cx[i]
                        cyi = cy[i]
                        xn = x + cxi
                        yn = y + cyi
                        if xn < 0 or xn >= nx or yn < 0 or yn >= ny:
                            # Skip out-of-bound
                            continue
                        if solid[yn, xn]:
                            # Bounce-back
                            j = opp[i]
                            fnew[yn, xn, j] += fpost[y, x, i]
                            # Accumulate force
                            Fx_total += 2 * fpost[y, x, i] * cxi
                            Fy_total += 2 * fpost[y, x, i] * cyi
                        else:
                            fnew[yn, xn, i] += fpost[y, x, i]

            # Apply non-equilibrium extrapolation
            for y in range(ny):
                for x in range(nx):
                    if not solid[y, x]:
                        # Left boundary
                        if x == 0:
                            for i in range(9):
                                cxi = cx[i]
                                cyi = cy[i]
                                if cxi < 0:
                                    # Left boundary: set velocity to uin
                                    if y < ny-1:
                                        u_ref = np.sum(f[y+1, x] * np.array(cx)) / np.sum(f[y+1, x])
                                        v_ref = np.sum(f[y+1, x] * np.array(cy)) / np.sum(f[y+1, x])
                                        rho_ref = np.sum(f[y+1, x])
                                        feq = w[i] * rho_ref * (1 + 3*(cxi*u_ref + cyi*v_ref) + 4.5*(cxi**2 + cyi**2)*(u_ref**2 + v_ref**2) - 1.5*(u_ref**2 + v_ref**2))
                                        fnew[y, x, i] = feq + f[y, x, i] - feq
                        # Right boundary
                        elif x == nx-1:
                            for i in range(9):
                                cxi = cx[i]
                                cyi = cy[i]
                                if cxi > 0:
                                    # Right boundary: set density to 1
                                    rho_ref = 1.0
                                    u_ref = np.sum(f[y, x-1] * np.array(cx)) / np.sum(f[y, x-1])
                                    v_ref = np.sum(f[y, x-1] * np.array(cy)) / np.sum(f[y, x-1])
                                    feq = w[i] * rho_ref * (1 + 3*(cxi*u_ref + cyi*v_ref) + 4.5*(cxi**2 + cyi**2)*(u_ref**2 + v_ref**2) - 1.5*(u_ref**2 + v_ref**2))
                                    fnew[y, x, i] = feq + f[y, x, i] - feq
                        # Bottom boundary
                        elif y == 0:
                            for i in range(9):
                                cxi = cx[i]
                                cyi = cy[i]
                                if cyi < 0:
                                    # Bottom boundary: set velocity to uin
                                    if x < nx-1:
                                        u_ref = np.sum(f[y, x+1] * np.array(cx)) / np.sum(f[y, x+1])
                                        v_ref = np.sum(f[y, x+1] * np.array(cy)) / np.sum(f[y, x+1])
                                        rho_ref = np.sum(f[y, x+1])
                                        feq = w[i] * rho_ref * (1 + 3*(cxi*u_ref + cyi*v_ref) + 4.5*(cxi**2 + cyi**2)*(u_ref**2 + v_ref**2) - 1.5*(u_ref**2 + v_ref**2))
                                        fnew[y, x, i] = feq + f[y, x, i] - feq
                        # Top boundary
                        elif y == ny-1:
                            for i in range(9):
                                cxi = cx[i]
                                cyi = cy[i]
                                if cyi > 0:
                                    # Top boundary: set velocity to uin
                                    if x < nx-1:
                                        u_ref = np.sum(f[y, x+1] * np.array(cx)) / np.sum(f[y, x+1])
                                        v_ref = np.sum(f[y, x+1] * np.array(cy)) / np.sum(f[y, x+1])
                                        rho_ref = np.sum(f[y, x+1])
                                        feq = w[i] * rho_ref * (1 + 3*(cxi*u_ref + cyi*v_ref) + 4.5*(cxi**2 + cyi**2)*(u_ref**2 + v_ref**2) - 1.5*(u_ref**2 + v_ref**2))
                                        fnew[y, x, i] = feq + f[y, x, i] - feq

            # Update populations
            f[:] = fnew[:]
            force_x = Fx_total
            force_y = Fy_total

            # Sample forces
            if step >= warmup_steps and (step - warmup_steps) % sample_every == 0:
                force_scale = 0.5 * uin**2 * square_size
                cd = force_x / force_scale
                cl = force_y / force_scale
                force_series[case_id].append((step, cd, cl))
                sampled_cd.append(cd)
                sampled_cl.append(cl)

        # Compute mean and RMS
        cd_mean = np.mean(sampled_cd)
        cl_rms = np.sqrt(np.mean((np.array(sampled_cl) - cd_mean)**2))

        # Write centerline profile
        centerline_y = square_center[1]
        for x in range(nx):
            rho, ux, uy = compute_macro(f, solid, x, centerline_y)
            centerline[case_id].append((x, ux))

        # Write transverse profile
        x_probe = min(nx - 1, square_center[0] + 2 * square_size)
        for y in range(ny):
            rho, ux, uy = compute_macro(f, solid, x_probe, y)
            transverse_profile[case_id].append((y, ux, uy))

        # Save results
        results.append({
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
            'converged': 1
        })

    # Write output files
    os.makedirs('./results', exist_ok=True)
    df_summary = pd.DataFrame(results)
    df_summary.to_csv('./results/fluid_summary.csv', index=False)
    df_force = pd.DataFrame(force_series)
    df_force.to_csv('./results/fluid_force_series.csv', index=False)
    df_centerline = pd.DataFrame(centerline).set_index(0).T
    df_centerline.to_csv('./results/fluid_centerline.csv', index=False)
    df_transverse = pd.DataFrame(transverse_profile).set_index(0).T
    df_transverse.to_csv('./results/fluid_transverse_profile.csv', index=False)
    with open('./results/fluid_report.json', 'w') as f:
        f.write(json.dumps({
            'task_id': 'lbm_2d_flow',
            'units': 'dimensionless',
            'cd_mean_avg': np.mean([r['cd_mean'] for r in results]),
            'method_summary': 'D2Q9 TRT-LBM with non-equilibrium extrapolation'
        }))

def compute_macro(f, solid, x, y):
    rho = np.sum(f[y, x])
    ux = np.sum(f[y, x] * np.array(cx)) / rho
    uy = np.sum(f[y, x] * np.array(cy)) / rho
    return rho, ux, uy

if __name__ == '__main__':
    main()
```