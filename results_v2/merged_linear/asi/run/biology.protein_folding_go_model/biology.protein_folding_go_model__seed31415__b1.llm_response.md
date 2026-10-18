```python
import numpy as np
import json
import scipy.interpolate

def load_parameters():
    with open('data/parameters.json') as f:
        params = json.load(f)
    return params

def compute_reference_contacts(ref_coords, s_min, r_c):
    N = len(ref_coords)
    contacts = []
    r_ref = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            if j - i < s_min:
                continue
            dist = np.linalg.norm(ref_coords[i] - ref_coords[j])
            if dist <= r_c:
                contacts.append((i, j))
                r_ref[i, j] = dist
                r_ref[j, i
    return np.array(contacts), r_ref

def compute_native_contacts(ref_coords, s_min, r_c):
    N = len(ref_coords)
    contacts = []
    r_ref = np.zeros((N, N=1))
```