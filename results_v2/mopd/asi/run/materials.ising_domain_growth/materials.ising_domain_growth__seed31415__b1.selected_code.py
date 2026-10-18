import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math

def read_parameters():
    with open('data/parameters.json') as f:
        params = json.load(f)
    return params

def read_initial_state():
    return np.load('data/initial_state.npy')

def write_results():
    # seriesA.npy
    np.save('series