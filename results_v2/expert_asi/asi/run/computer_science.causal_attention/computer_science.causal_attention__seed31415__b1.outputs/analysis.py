import numpy as np
import os
import json

def sequence_operator_forward(x, segment_ids, params, meta, return_cache=False):
    # Resolve roles from params
    # Implement forward pass as per definition
    # Return output and cache if needed

def sequence_operator_backward(sensitivity, cache, params, meta):
    # Compute gradients for each parameter
    # Return gradient dicts

def sequence_operator_hvp(x, segment_ids, params, meta, sensitivity, direction):
    # Compute Hessian-vector product
    # Return HVP dicts

# Load inputs
x = np.load('data/x.npy')
segment_ids = np.load('data/segment_ids.npy')
sensitivity = np.load('data/sensitivity.npy')
meta = json.load(open('data/meta.json'))

# Compute forward pass
output = sequence_operator_forward(x, segment_ids, params, meta, return_cache=True)[0]

# Compute gradients
grads = sequence_operator_backward(sensitivity, cache, params, meta)

# Compute HVP
hvp = sequence_operator_hvp(x, segment_ids, params, meta, sensitivity, direction)

# Save outputs
np.save('results/output.npy', output)
np.save('results/input_gradient.npy', grads['x'])
np.save('results/input_hvp.npy', hvp['x'])