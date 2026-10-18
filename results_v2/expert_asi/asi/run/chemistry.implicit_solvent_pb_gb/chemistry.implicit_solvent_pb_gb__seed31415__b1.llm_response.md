```python
import numpy as np
import json
import os
import matplotlib.pyplot as plt

def load_case(case_idx, manifest):
    case = manifest['cases'][case_idx]
    grid = case['grid']
    entity_groups = case['entity_groups']
    binding_expr = case['binding_expr']
    field_entity = case['field_entity']
    dielectric = case['dielectric']
    screening_kappa = case['screening_kappa']
    probe_radius = case['probe_radius']
    response_regime = case['response_regime']
    return case

def read_volume(path):
    return np.load(path).astype(np.float32)

def compute_pb_field(grid, charges, dielectric, screening_kappa, spacing, inner_dielectric=1.0, outer_dielectric=80.0, linearized=True):
    # Implement finite-volume PB solver here
    pass

def compute_born_cross_term(part_a_sites, part_b_sites, dielectric, screening_kappa, probe_radius):
    # Implement Born cross-term approximation here
    pass

def main():
    with open('data/manifest.json') as f:
        manifest = json.load(f)
    volumes = [read_volume('data/volume_a.npy'), read_volume('data/volume_b.npy')]
    queries = np.load('data/queries.npy')
    plane_spec = json.load('data/plane_spec.json')
    
    results = []
    field_samples = np.zeros((8, 24), dtype=np.float32)
    field_slice = np.zeros((8, 35, 35), dtype=np.float32)
    
    for case_idx in range(8):
        case = load_case(case_idx, manifest)
        grid = case['grid']
        entity_groups = case['entity_groups']
        binding_expr = case['binding_expr']
        field_entity = case['field_entity']
        dielectric = case['dielectric']
        screening_kappa = case['screening_kappa']
        probe_radius = case['probe_radius']
        response_regime = case['response_regime']
        
        # Compute primary metric
        primary = compute_pb_binding(binding_expr, entity_groups, volumes, case, grid, probe_radius, response_regime)
        
        # Compute secondary metric
        part_a_sites = get_sites(entity_groups['part_a'], 'data/sites.csv')
        part_b_sites = get_sites(entity_groups['part_b'], 'data/sites.csv')
        secondary = compute_born_cross_term(part_a_sites, part_b_sites, dielectric, screening_kappa, probe_radius)
        
        # Compute field samples and slice
        field_samples[case_idx] = compute_field_samples(grid, queries, volumes, case, probe_radius)
        field_slice[case_idx] = compute_field_slice(grid, plane_spec, volumes, case, probe_radius)
        
        results.append({
            'case_id': case_idx,
            'metric_primary_kcal_mol': primary,
            'metric_secondary_kcal_mol': secondary,
            'metric_gap_kcal_mol': primary - secondary
        })
    
    np.savetxt('results/summary_metrics.csv', results, delimiter=',')
    np.save('results/field_samples.npy', field_samples)
    np.save('results/field_slice.npy', field_slice)
    plt.imshow(field_slice[0], cmap='viridis')
    plt.savefig('results/overview.png')
```