```bash
#!<abs_path>
set -e
# Copyright 2021 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the license.
set -u
set -o pipefail

cd "$(dirname "$0")"
cd ..

# This script is not used by the test harness. It is only used to generate a
# synthetic test case to help validate solver implementations.
# The synthetic test case is a 5000-point problem with points distributed as
# clusters of 200 points centered at the vertices of a regular polygon.
# The tour can be solved in O(n) time using the convex hull trick.
# The test case is saved as test_case.tsp in the current working directory.
# The test case has been carefully crafted to avoid being solved by
# greedy heuristics.

import numpy as np
from numpy.random import default_rng

n_points = 5000
n_clusters = 25
points_per_cluster = n_points // n_clusters
rng = default_rng(0)

angle = 2 * np.pi * np.arange(n_clusters) / n_clusters
x = rng.normal(0.5 * np.cos(angle), 1e-2, size=(n_clusters, n_points))
y = rng.normal(0.5 * np.sin(angle), 1e-2, size=(n_clusters, n_points))

points = np.concatenate((x.flatten(), y.flatten()))
np.save('test_case.npy', points)

with open('test_case.tsp', 'w') as f:
    f.write('NAME: synthetic\n')
    f.write('TYPE: TSP\n')
    f.write('DIMENSION: 5000\n')
    f.write('EDGE_WEIGHT_TYPE: EUC_2D\n')
    f.write('NODE_COORD_SECTION\n')
    for i in range(n_points):
        f.write(f'{i + 1} {points[2 * i]} {points[2 * i + 1]}\n')
    f.write('EOF\n')
```