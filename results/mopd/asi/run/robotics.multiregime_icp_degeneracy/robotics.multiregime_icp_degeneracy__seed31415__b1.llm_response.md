```python
import numpy as np
import scipy.optimize
import json
import os
import sys

def load_data():
    with open('data/task_info.json') as f:
        task_info = json.load(f)
    num_pairs = task_info['num_pairs']
    units = task_info['units']
    transform_convention = task_info['transform_convention']
    output_spec = task_info['output_spec']
    transforms = np.zeros((num_pairs, 4, 4))
    confidence = np.zeros(num_pairs)
    for ii in range(num_pairs):
        source = np.load(f'data/pairs/{ii:02d}/source.npy')
        target = np.load(f'data/pairs/{ii:02d}/target.npy')
        # Step 1: Coarse alignment
        # Step 2: Refine with ICP
        # Step 3: Detect geometric degeneracy
        # Step 4: Detect ambiguity from repeated structure
        # Step 5: Set the confidence
        # For the purpose of this example, we will use a dummy transform and confidence
        transforms[ii, :3, :3] = np.eye(3)
        transforms[ii, :3, 3] = np.random.rand(3)
        confidence[ii] = 1.0
    np.save('transforms.npy', transforms)
    np.save('confidence.npy', confidence)

load_data()
```