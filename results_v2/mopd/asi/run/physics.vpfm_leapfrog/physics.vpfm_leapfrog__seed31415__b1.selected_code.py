import numpy as np
import sys
import os
import time
import scipy.fft
import scipy.sparse
import scipy.sparse.linalg

# Constants
N = 135
dx = 1.0 / N
inv_dx = N
dt_frame = 0.1
CFL = 1.8797
dt = 0.1
sub_t = 0.0
num_frames = 108
vorticity_frames = []
kinetic_energy = []

# Load initial conditions
u_x