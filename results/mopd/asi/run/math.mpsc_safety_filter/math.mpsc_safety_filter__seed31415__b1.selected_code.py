import numpy as np
import scipy.optimize as opt
import json

# Load system data
with open('data/system.json') as f:
    system_data = json.load(f)
A = np.array(system_data['A'])
B = np.array(system_data['B'])
K = np.array(system_data['K'])
disturbance_scenarios = np.array(system_data['disturbance_scenarios'])
p_floor = system_data['p_floor']
p_ceiling = system_data['p_ceiling']
tau_bounds = system_data['scenario_design']['tau_bounds']
action_metric = np.array(system_data['solver_regularization']['action_metric'])
rho_z = system_data['solver_regularization']['rho_z']
rho_v = system_data['solver_regularization']['rho_v']

# Ellipsoid design
def objective(P, tau):
    return -np.log(np.linalg.det(P))

def constraint(P, tau, w_i):
    A_K = A + B @ K
    left = np.block([[A_K.T @ P @ A_K - tau * P, A_K.T @ P @ w_i],
                    [w_i.T @ P @ A_K, w_i.T @ P @ w_i + tau - 1]])
    return np.max(np.linalg.eigvals(left)) <= 0

def optimize_ellipsoid():
    best_P = None
    best_tau = None
    best_objective = float('inf')
    for tau in np.linspace(*tau_bounds, 1000):
        result = opt.minimize(lambda P: objective(P, tau), x0=np.eye(A.shape[0]), bounds=[(p_floor, p_ceiling)]*A.shape[0])
        P = result.x
        if constraint(P, tau, disturbance_scenarios[0]):
            current_objective = objective(P, tau)
            if current_objective < best_objective:
                best_P = P
                best_tau = tau
                best_objective = current_objective
    return best_P, best_tau

P, tau = optimize_ellipsoid()

# Support values
support_state = np.sqrt(np.diag(1 / P))
support_input = np.sqrt(np.diag(1 / (P @ K.T @ K)))

# Terminal set
terminal_candidate_library = json.load(open('data/public_cases.json'))
terminal_backup = terminal_candidate_library['terminal_backup']
H_x = np.array(terminal_backup['H'])
h_x = np.array(terminal_backup['h'])
H_u = np.array(terminal_backup['H_u'])
h_u = np.array(terminal_backup['h_u'])

def build_terminal_set():
    accepted_states = []
    for state in terminal_candidate_library['terminal_candidate_library']['states']:
        if np.all(H_x @ state <= h_x):
            accepted_states.append(state)
    while True:
        new_states = []
        for state in terminal_candidate_library['terminal_candidate_library']['states']:
            if np.all(H_x @ state <= h_x) and np.all(H_u @ state <= h_u):
                if np.all(A @ state + B @ state <= np.max(accepted_states, axis=0)):
                    new_states.append(state)
        if not new_states:
            break
        accepted_states.extend(new_states)
    hull = ConvexHull(accepted_states)
    H_f = hull.equations[:, :H_x.shape[1]]
    h_f = hull.equations[:, -1]
    return H_f, h_f

H_f, h_f = build_terminal_set()

# Write certificate
certificate = {
    "schema_version": 2,
    "error_ellipsoid": {"P": P.tolist(), "tau": tau},
    "support_values": {"state": support_state.tolist(), "input": support_input.tolist()},
    "tightened_polytopes": {
        "state": {"H": H_x.tolist(), "h": h_x.tolist()},
        "input": {"H": H_u.tolist(), "h": h_u.tolist()}
    },
    "terminal_polytope": {"H": H_f.tolist(), "h": h_f.tolist()}
}
with open('certificate.json', 'w') as f:
    json.dump(certificate, f)

# Online filter
def make_filter(task_data, config=None):
    def filter_control(x, u_learning, t=0, memory=None):
        # Solve the SOCP
        pass
    return filter_control