```python
import numpy as np
import scipy.linalg as la
from scipy.spatial import ConvexHull
import json

def make_filter(task_data, config=None):
    class SafetyFilter:
        def __init__(self, P, H_x, h_x, H_u, h_u, X_f, K):
            self.P = P
            self.H_x = H_x
            self.h_x = h_x
            self.H_u = H_u
            self.h_u = h_u
            self.X_f = X_f
            self.K = K

        def filter_control(self, x, u_learning, t=0, memory=None):
            # Solve the convex SOCP problem
            # Placeholder for actual optimization implementation
            # This is a simplified example; actual implementation would use a solver like CVXPY
            v = np.linalg.lstsq(self.H_u, self.h_u - self.K @ x, rcond=None)[0]
            return v

    # Load necessary parameters from task_data
    A = task_data['A']
    B = task_data['B']
    K = task_data['K']
    H_x = task_data['H_x']
    h_x = task_data['h_x']
    H_u = task_data['H_u']
    h_u = task_data['h_u']
    X_f = task_data['X_f']

    # Construct the SafetyFilter instance
    return SafetyFilter(task_data['P'], H_x, h_x, H_u, h_u, X_f, K)

def certificate_to_json(schema_version, error_ellipsoid, support_values, tightened_polytopes, terminal_polytope):
    with open('certificate.json', 'w') as f:
        json.dump({
            'schema_version': schema_version,
            'error_ellipsoid': error_ellipsoid,
            'support_values': support_values,
            'tightened_polytopes': tightened_polytopes,
            'terminal_polytope': terminal_polytope
        }, f)

def main():
    # Load system data and public cases
    import json
    with open('data/system.json', 'r') as f:
        system_data = json.load(f)
    with open('data/public_cases.json', 'r') as f:
        public_cases = json.load(f)

    # Ellipsoid design
    A = system_data['A']
    B = system_data['B']
    K = system_data['K']
    A_K = A + B @ K
    disturbance_scenarios = public_cases['disturbance_scenarios']
    tau_bounds = public_cases['scenario_design']['tau_bounds']

    # Perform one-dimensional search over tau
    best_tau = 0.0
    best_P = None
    best_support_values = None
    for tau in np.linspace(tau_bounds[0], tau_bounds[1], 100):
        # Solve the LMI problem for P
        # Placeholder for actual LMI solver
        P = np.eye(A.shape[0]) * 1.0
        # Compute support values
        support_state = np.sqrt(np.diag(P))
        support_input = np.sqrt(np.diag(K @ P @ K.T))
        # Validate eigenvalue residuals
        # Placeholder for actual validation
        if validate_eigenvalues(P):
            best_tau = tau
            best_P = P
            best_support_values = {'state': support_state, 'input': support_input}
            break

    # Terminal set construction
    terminal_candidate_library = public_cases['terminal_candidate_library']
    terminal_backup = public_cases['terminal_backup']
    core_polytope_vertices = terminal_backup['core_polytope_vertices']
    accepted_states = []
    accepted_controls = []
    # Process candidates in batches
    for candidate in terminal_candidate_library['states']:
        if candidate in core_polytope_vertices:
            accepted_states.append(candidate)
    # Convert to H_f z <= h_f
    # Placeholder for actual convex hull computation
    H_f = np.array([[1., 0.], [0., 1.]])
    h_f = np.array([1., 1.])

    # Write certificate
    certificate_to_json(
        schema_version=2,
        error_ellipsoid={'P': best_P.tolist(), 'tau': best_tau},
        support_values=best_support_values,
        tightened_polytopes={
            'state': {'H': H_x.tolist(), 'h': h_x.tolist()},
            'input': {'H': H_u.tolist(), 'h': h_u.tolist()}
        },
        terminal_polytope={'H': H_f.tolist(), 'h': h_f.tolist()}
    )

if __name__ == '__main__':
    main()
```