import json
import numpy as np
import scipy.optimize as opt
import scipy.spatial as spatial
import cvxpy as cp
from scipy.linalg import eigvals

# Load data
with open('data/system.json') as f:
    system_data = json.load(f)

with open('data/public_cases.json') as f:
    public_cases = json.load(f)

# Extract system matrices
A = np.array(system_data['A'])
B = np.array(system_data['B'])
K = np.array(system_data['K'])
disturbance_scenarios = np.array(public_cases['disturbance_scenarios'])
tau_bounds = np.array(public_cases['scenario_design']['tau_bounds'])
p_floor = public_cases['scenario_design']['p_floor']
p_ceiling = public_cases['scenario_design']['p_ceiling']
H_x = np.array(public_cases['H_x'])
h_x = np.array(public_cases['h_x'])
H_u = np.array(public_cases['H_u'])
h_u = np.array(public_cases['h_u'])
terminal_candidate_library = public_cases['terminal_candidate_library']
terminal_backup = public_cases['terminal_backup']
action_metric = np.array(public_cases['solver_regularization']['action_metric'])
rho_z = public_cases['solver_regularization']['rho_z']
rho_v = public_cases['solver_regularization']['rho_v']

# Ellipsoid Design
def compute_ellipsoid(tau):
    # Objective: maximize logdet(P)
    # Constraints: P >= p_floor I, P <= p_ceiling I, and the matrix inequalities
    # Initialize P
    P = np.eye(A.shape[0])
    # Solve optimization problem
    # Convert to CVXPY problem
    P_var = cp.Variable((A.shape[0], A.shape[0]), symmetric=True)
    constraints = [P_var >= p_floor * np.eye(A.shape[0]), P_var <= p_ceiling * np.eye(A.shape[0])]
    for w in disturbance_scenarios:
        # Construct the matrix inequality
        A_K = A + B @ K
        left = cp.bmat([[A_K.T @ P_var @ A_K - tau * P_var, A_K.T @ P_var @ w],
                        [w.T @ P_var @ A_K, w.T @ P_var @ w + tau - 1]])
        constraints.append(left <= 0)
    objective = cp.Maximize(cp.log_det(P_var))
    prob = cp.Problem(objective, constraints)
    prob.solve()
    if prob.status != 'OPTIMAL':
        return None, None
    P_opt = P_var.value
    # Check eigenvalues
    eigenvalues = eigvals(P_opt)
    if np.any(eigenvalues < p_floor) or np.any(eigenvalues > p_ceiling):
        return None, None
    return P_opt, tau

# Perform one-dimensional search over tau
tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], 100)
best_P = None
best_tau = None
for tau in tau_grid:
    P, _ = compute_ellipsoid(tau)
    if P is not None:
        best_P = P
        best_tau = tau

# Compute support values
support_state = np.sqrt(np.diag(best_P))
support_input = np.sqrt(np.diag(best_P @ K @ K.T))

# Data-Driven Terminal Set
# Extract candidate states and controls
states = np.array(terminal_candidate_library['states'])
controls = np.array(terminal_candidate_library['controls'])
core_polytope_vertices = np.array(terminal_backup['core_polytope'])

# Start with states that match core polytope vertices
accepted_states = []
accepted_controls = []
for state in states:
    if np.allclose(state, core_polytope_vertices):
        accepted_states.append(state)
        # Find corresponding control
        idx = np.argmin(np.linalg.norm(controls - state, axis=1))
        accepted_controls.append(controls[idx])

# Build convex hull
hull = spatial.ConvexHull(accepted_states)
current_hull = accepted_states

# Iteratively add states that satisfy constraints
while True:
    new_states = []
    new_controls = []
    for i in range(len(states)):
        state = states[i]
        control = controls[i]
        if np.all(H_x @ state <= h_x - support_state) and np.all(H_u @ control <= h_u - support_input):
            # Check if state is in convex hull of current accepted states
            if np.allclose(state, current_hull):
                new_states.append(state)
                new_controls.append(control)
            else:
                # Check if state is in convex hull of current accepted states
                if np.allclose(state, current_hull):
                    new_states.append(state)
                    new_controls.append(control)
    if not new_states:
        break
    current_hull = np.vstack((current_hull, new_states))
    accepted_states.extend(new_states)
    accepted_controls.extend(new_controls)

# Convert to H_f z <= h_f
hull = spatial.ConvexHull(accepted_states)
H_f = np.array(hull.equations[:, :hull.dims])
h_f = np.array(hull.equations[:, -1])

# Save certificate
certificate = {
    "schema_version": 2,
    "error_ellipsoid": {"P": best_P.tolist(), "tau": best_tau},
    "support_values": {"state": support_state.tolist(), "input": support_input.tolist()},
    "tightened_polytopes": {
        "state": {"H": H_x.tolist(), "h": (h_x - support_state).tolist()},
        "input": {"H": H_u.tolist(), "h": (h_u - support_input).tolist()}
    },
    "terminal_polytope": {"H": H_f.tolist(), "h": h_f.tolist()}
}

with open('certificate.json', 'w') as f:
    json.dump(certificate, f)

# Online Filter
def make_filter(task_data, config=None):
    class SafetyFilter:
        def filter_control(self, x, u_learning, t=0, memory=None):
            # Define variables
            z = cp.Variable((A.shape[0], 1))
            v = cp.Variable((B.shape[1], N))
            # Constraints
            constraints = []
            for i in range(N):
                constraints.append(cp.bmat([[z, v[i]], [v[i], cp.Constant(0)]]))
                constraints.append(H_x @ z <= h_x - support_state)
                constraints.append(H_u @ v[i] <= h_u - support_input)
            constraints.append((x - z) @ P @ (x - z) <= 1)
            constraints.append(z[N] in X_f)
            # Objective
            objective = cp.Minimize((u_safe - u_learning) @ action_metric @ (u_safe - u_learning) + rho_z * cp.norm(z - x, 2)**2 + rho_v * cp.sum(cp.norm(v, 2)**2))
            # Solve
            prob = cp.Problem(objective, constraints)
            prob.solve()
            if prob.status != 'OPTIMAL':
                return np.zeros(B.shape[1])
            u_safe = v[0] + K @ (x - z)
            return u_safe.value
    return SafetyFilter()