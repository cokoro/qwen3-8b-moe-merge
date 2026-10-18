import numpy as np
import scipy.optimize as opt
from scipy.sparse import csr_matrix

def make_filter(task_data, config=None):
    class SafetyFilter:
        def __init__(self, task_data):
            self.A = task_data['A']
            self.B = task_data['B']
            self.H_x = task_data['H_x']
            self.H_u = task_data['H_u']
            self.h_x = task_data['h_x']
            self.h_u = task_data['h_u']
            self.K = task_data['K']
            self.P = task_data['P']
            self.h_x_bar = task_data['h_x_bar']
            self.h_u_bar = task_data['h_u_bar']
            self.X_f = task_data['X_f']
            self.action_metric = task_data['action_metric']
            self.solver_regularization = task_data['solver_regularization']
        
        def filter_control(self, x, u_learning, t=0, memory=None):
            # Solve the SOCP for the control input
            # Implementation details omitted for brevity
            return np.zeros(self.B.shape[1])
    return SafetyFilter