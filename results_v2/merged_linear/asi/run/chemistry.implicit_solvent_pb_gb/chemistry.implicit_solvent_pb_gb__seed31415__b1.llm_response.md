```python
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load manifest
with open('data/manifest.json') as f:
    manifest = json.load(f)

# Load sites
sites = pd.read_csv('data/sites.csv')

# Load volumes
volume_a = np.load('data/volume_a.npy')
volume_b = np.load('data/volume_b.npy')

# Load queries
queries = np.load('data/queries.npy')

# Load plane spec
with open('data/plane_spec.json') as f:
    plane_spec = json.load(f)

# Initialize output arrays
summary_metrics = []
field_samples = []
field_slices = []

# Process each case
for case_idx in range(8):
    case = manifest['cases'][case_idx]
    grid = case['grid']
    response_regime = case['response_regime']
    entity_groups = case['entity_groups']
    binding_expression = case['binding_expression']
    field_entity = case['field_entity']
    dielectric = case['dielectric']
    screening_kappa = case['screening_kappa']
    probe_radius = case['probe_radius']
    inner_dielectric = case['inner_dielectric']
    outer_dielectric = case['outer_dielectric']
    temp = case['temp']
    ionic_strength = case['ionic_strength']
    site_table = sites[site_table]

    # Reconstruct entity surfaces
    entity_masks = {}
    for entity_name, entity_rows in entity_groups.items():
        entity_atoms = site_table[entity_rows]
        entity_radii = entity_atoms['radius'].values
        entity_coords = entity_atoms[['x', 'y', 'z']].values
        entity_mask = np.zeros_like(volume_a, dtype=bool)
        for atom_idx in range(entity_radii.shape[0]):
            atom_radius = entity_radii[atom_idx]
            atom_coords = entity_coords[atom_idx]
            entity_mask |= distance_transform(volume_a, atom_coords, atom_radius)

        entity_masks[entity_name] = entity_mask

    # Solve PB for each entity
    entity_potentials = {}
    for entity_name, entity_mask in entity_masks.items():
        entity_volume = volume_a if entity_name == field_entity else entity_mask
        entity_potential = solve_pb(grid, entity_volume, entity_mask, response_regime, dielectric, screening_kappa, probe_radius, inner_dielectric, outer_dielectric, temp, ionic_strength)
        entity_potentials[entity_name] = entity_potential

    # Compute primary metric
    primary_metric = evaluate_binding_expression(binding_expression, entity_potentials)

    # Compute secondary metric
    part_a_sites = entity_masks['part_a']
    part_b_sites = entity_masks['part_b']
    part_a_coords = site_table[part_a_sites[['x', 'y', 'z']].values]
    part_b_coords = site_table[part_b_sites[['x', 'y', 'z']].values]
    part_a_radii = part_a_sites['radius'].values
    part_b_radii = part_b_sites['radius'].values

    secondary_metric = compute_born_cross_term(part_a_coords, part_a_radii, part_b_coords, part_b_radii, inner_dielectric, outer_dielectric, screening_kappa, probe_radius)

    # Compute field samples
    field_samples_case = evaluate_field_samples(grid, entity_potentials, queries, part_a_sites, part_b_sites, inner_dielectric, outer_dielectric, screening_kappa, probe_radius)
    field_samples.append(field_samples_case)

    # Compute field slice
    field_slice_case = evaluate_field_slice(grid, entity_potentials, plane_spec, part_a_sites, part_b_sites, inner_dielectric, outer_dielectric, screening_kappa, probe_radius)
    field_slices.append(field_slice_case)

    # Save metrics
    summary_metrics.append({
        'case_id': case_idx,
        'metric_primary_kcal_mol': primary_metric,
        'metric_secondary_kcal_mol': secondary_metric,
        'metric_gap_kcal_mol': primary_metric - secondary_metric
    })

# Save outputs
summary_metrics_df = pd.DataFrame(summary_metrics)
summary_metrics_df.to_csv('results/summary_metrics.csv', index=False)
np.save('results/field_samples.npy', np.array(field_samples, dtype=np.float32))
np.save('results/field_slice.npy', np.array(field_slices, dtype=np.float32))

# Generate overview plot
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    axes[i // 4, i % 4].imshow(field_slices[i], cmap='viridis')
plt.tight_layout()
plt.savefig('results/overview.png')
```