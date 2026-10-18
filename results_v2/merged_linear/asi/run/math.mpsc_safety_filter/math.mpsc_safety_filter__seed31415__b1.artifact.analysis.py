import json
import numpy as np
import cvxpy as cp
from scipy.spatial import ConvexHull

def make_filter(task_data, config=None):
    config = config or dict()
    A = np.array(task_data["A"])
    B = np.array(task_data["B"])
    H_x = np.array(task_data["H_x"])
    h_x = np.array(task_data["h_x"])
    H_u = np.array(task_data["H_u"])
    h_u = np.array(task_data["h_u"])
    terminal_candidate_library = task_data["terminal_candidate_library"]
    terminal_backup = task_data["terminal_backup"]
    solver_regularization = config.get("solver_regularization", dict())

    # Ellipsoid Design
    K = np.array(config["error_feedback_gain"])
    A_K = A + B @ K
    disturbance_scenarios = np.array(config["disturbance_scenarios"])
    tau_bounds = config["scenario_design"]["tau_bounds"]
    p_floor = config["scenario_design"]["p_floor"]
    p_ceiling = config["scenario_design"]["p_ceiling"]
    N = config["public"]["horizon"]

    # One-dimensional search over tau
    tau_grid = np.linspace(tau_bounds[0], tau_bounds[1], 1000)
    best_P = None
    best_tau = None
    best_logdet = -np.inf

    for tau in tau_grid:
        # Objective: maximize logdet(P)
        P = cp.Variable((A.shape[0], A.shape[0]), symmetric=True)
        constraints = [
            cp.bmat([
                [A_K.T @ P @ A_K - tau * P, A_K.T @ P @ disturbance_scenarios[i], ],
                [disturbance_scenarios[i].T @ P @ A_K, disturbance_scenarios[i].T @ P @ disturbance_scenarios[i] + tau - 1]
            ]) << 0 for i in range(disturbance_scenarios.shape[0])
        ]
        constraints += [P >= p_floor * np.eye(A.shape[0]), P <= p_ceiling * np.eye(A.shape[0])]
        obj = cp.Maximize(cp.log_det(P))
        prob = cp.Problem(obj, constraints)
        prob.solve(solver='SCS', eps=1e-6, verbose=False)

        if prob.status == 'optimal':
            if prob.value > best_logdet:
                best_P = P.value
                best_tau = tau
                best_logdet = prob.value

    P = best_P
    support_values = np.sqrt(np.diag(np.linalg.inv(P)))

    # Terminal Set Construction
    accepted_states = []
    for s_j, q_j in terminal_candidate_library["states"]:
        if np.all(H_x @ s_j <= h_x + support_values):
            accepted_states.append(s_j)

    accepted_states = np.array(accepted_states)
    accepted_controls = np.array([q_j for s_j, q_j in terminal_candidate_library["states"] if np.all(H_x @ s_j <= h_x + support_values)])

    for _ in range(10):
        new_states = []
        for s_j, q_j in terminal_candidate_library["states"]:
            if np.all(H_x @ s_j <= h_x + support_values) and np.all(H_u @ q_j <= h_u + support_values):
                A_s = A @ s_j + B @ q_j
                if np.all(np.dot(A_s, np.linalg.inv(accepted_states.T @ accepted_states)) <= np.ones(accepted_states.shape[0])):
                    new_states.append(s_j)
        new_states = np.array(new_states)
        if new_states.shape[0] == 0:
            break
        accepted_states = np.vstack((accepted_states, new_states))
        accepted_controls = np.vstack((accepted_controls, np.array([q_j for s_j, q_j in terminal_candidate_library["states"] if s_j in new_states])))
    accepted_states = np.unique(accepted_states, axis=0)
    accepted_controls = np.unique(accepted_controls, axis=0)

    hull = ConvexHull(accepted_states)
    X_f = np.vstack((accepted_states, accepted_controls))
    H_f = np.hstack((H_x, H_u))
    h_f = np.hstack((h_x, h_u))

    # Online Filter
    class SafetyFilter:
        def __init__(self, P, K, H_x, h_x, H_u, h_u, X_f, H_f, h_f, A, B, N, solver_regularization):
            self.P = P
            self.K = K
            self.H_x = H_x
            self.h_x = h_x
            self.H_u = H_u
            self.h_u = h_u
            self.X_f = X_f
            self.H_f = H_f
            self.h_f = h_f
            self.A = A
            self.B = B
            self.N = N
            self.rho_z = solver_regularization.get("rho_z", 1e-3)
            self.rho_v = solver_regularization.get("rho_v", 1e-3)
            self.action_metric = solver_regularization.get("action_metric", np.eye(B.shape[1]))

        def filter_control(self, x, u_learning, t=0, memory=None):
            z_0 = x
            v = np.zeros(self.B.shape[1])
            for i in range(self.N):
                z_i = self.A @ z_0 + self.B @ v
                if np.all(self.H_x @ z_i <= self.h_x + self.support_values):
                    pass
                else:
                    break
                z_0 = z_i
                v = np.zeros(self.B.shape[1])
            u_safe = v + self.K @ (x - z_0)
            return u_safe

    return SafetyFilter(P, K, H_x, h_x, H_u, h_u, X_f, H_f, h_f, A, B, N, solver_regularization)