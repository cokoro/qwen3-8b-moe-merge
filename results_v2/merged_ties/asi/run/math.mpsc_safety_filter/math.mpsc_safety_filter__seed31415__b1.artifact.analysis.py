```bash
import numpy as np
import json
from scipy import linalg
from scipy.optimize import Bounds
import cvxpy as cp

def run():
    with open('data/system.json') as f:
        system = json.load(f)
    A = np.array(system['A'])
    B = np.array(system['B'])
    error_feedback_gain = np.array(system['error_feedback_gain'])
    disturbance_scenarios = np.array(system['disturbance_scenarios'])
    p_floor = system['p_floor']
    p_ceiling = system['p_ceiling']
    tau_bounds = system['scenario_design']['tau_bounds']
    terminal_candidate_library = json.load(open('data/public_cases.json'))

    K = error_feedback_gain
    A_K = A + B @ K

    # Scenario-based ellipsoidal MPSC filter design
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], 500)
    best_logdet = -np.inf
    best_P = None
    for tau in tau_grid:
        P = cp.Variable((A.shape[0], A.shape[0]), PSD=True)
        obj = cp.Maximize(cp.log_det(P))
        constr = [
            P >> p_floor * np.eye(A.shape[0]),
            P << p_ceiling * np.eye(A.shape[0]),
        ]
        for w in disturbance_scenarios:
            constr += [
                cp.bmat([
                    [A_K.T @ P @ A_K - tau * P, A_K.T @ P @ w.reshape(-1, 1)],
                    [w.reshape(1, -1) @ P @ A_K, w.reshape(1, -1) @ P @ w.reshape(-1, 1) + tau - 1]
                ]) << 0
            ]
        prob = cp.Problem(obj, constr)
        prob.solve(solver='SCS', verbose=False)
        if prob.status == 'optimal':
            logdet = prob.value
            if logdet > best_logdet:
                best_logdet = logdet
                best_P = P.value

    P = best_P
    tau = tau_grid[np.argmin(np.abs(tau_grid - tau))]
    support_state = np.sqrt(np.diag(P))
    support_input = np.sqrt(np.diag(K.T @ P @ K))

    # Terminal set enlargement
    accepted_states = []
    accepted_controls = []
    for candidate in terminal_candidate_library['states']:
        s = np.array(candidate)
        q = np.array(terminal_candidate_library['controls'][terminal_candidate_library['states'].index(candidate)])
        if np.all(H_x @ s <= h_x_bar) and np.all(H_u @ q <= h_u_bar) and np.all(H_x @ (A @ s + B @ q) <= h_x_bar):
            accepted_states.append(s)
            accepted_controls.append(q)
    while True:
        new_accepted_states = []
        new_accepted_controls = []
        for candidate in terminal_candidate_library['states']:
            s = np.array(candidate)
            q = np.array(terminal_candidate_library['controls'][terminal_candidate_library['states'].index(candidate)])
            if np.all(H_x @ s <= h_x_bar) and np.all(H_u @ q <= h_u_bar) and np.all(H_x @ (A @ s + B @ q) <= h_x_bar):
                new_accepted_states.append(s)
                new_accepted_controls.append(q)
        if not new_accepted_states:
            break
        accepted_states = new_accepted_states
        accepted_controls = new_accepted_controls

    H_f = np.array([h_f_i for h_f_i in terminal_candidate_library['terminal_polytope']['H']])
    h_f = np.array(terminal_candidate_library['terminal_polytope']['h'])

    with open('certificate.json', 'w') as f:
        json.dump({
            'schema_version': 2,
            'error_ellipsoid': {'P': best_P.tolist(), 'tau': tau},
            'support_values': {'state': support_state.tolist(), 'input': support_input.tolist()},
            'tightened_polytopes': {
                'state': {'H': H_x.tolist(), 'h': h_x_bar.tolist()},
                'input': {'H': H_u.tolist(), 'h': h_u_bar.tolist()}
            },
            'terminal_polytope': {'H': H_f.tolist(), 'h': h_f.tolist()}
        }, f)

    with open('analysis.py', 'w') as f:
        f.write('''
import numpy as np
from scipy.optimize import Bounds
import cvxpy as cp

def make_filter(task_data, config=None):
    A = np.array(task_data['A'])
    B = np.array(task_data['B'])
    H_x = np.array(task_data['H_x'])
    h_x = np.array(task_data['h_x'])
    H_u = np.array(task_data['H_u'])
    h_u = np.array(task_data['h_u'])
    P = np.array(task_data['P'])
    K = np.array(task_data['K'])
    H_f = np.array(task_data['H_f'])
    h_f = np.array(task_data['h_f'])

    def filter_control(x, u_learning, t=0, memory=None):
        z = cp.Variable((A.shape[0], 1))
        v = cp.Variable((B.shape[1], N))
        obj = cp.Minimize(cp.sum_squares(u_learning - K @ (x - z)))
        for i in range(N):
            obj += cp.sum_squares(v[:, i])
        for i in range(N):
            obj += cp.sum_squares(z[:, i] - A @ z[:, i-1] - B @ v[:, i])
        for i in range(N):
            obj += cp.sum_squares(H_x @ z[:, i] - h_x)
        for i in range(N):
            obj += cp.sum_squares(H_u @ v[:, i] - h_u)
        obj += cp.sum_squares(z[:, 0] - x)
        prob = cp.Problem(obj)
        prob.solve(solver='SCS', verbose=False)
        u_safe = K @ (x - z[:, 0])
        return u_safe
''')
```