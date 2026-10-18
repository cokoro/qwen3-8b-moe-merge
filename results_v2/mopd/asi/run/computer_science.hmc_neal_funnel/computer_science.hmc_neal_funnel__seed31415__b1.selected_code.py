import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Read transition schedule
with open('data/transition_schedule.json') as f:
    schedule = json.load(f)
n_chains = schedule['n_chains']
dimension = schedule['dimension']
warmup_transitions = schedule['warmup_transitions']
kept_transitions = schedule['kept_transitions']
total_transitions = schedule['total_transitions']
step_size = schedule['step_size']
integration_steps = schedule['integration_steps']
audit_transitions = schedule['audit_transitions']
holdout_chains = schedule['holdout_chains']
holdout_transitions = schedule['holdout_transitions']
holdout_step_size = schedule['holdout_step_size']
holdout_integration_steps = schedule['holdout_integration_steps']
holdout_audit_transitions = schedule['holdout_audit_transitions']
holdout_audit_transitions = schedule['holdout_audit_transitions']

# Read target manifest
with open('data/target_manifest.json') as f:
    manifest = json.load(f)
matrix = manifest['sampler_chart']['matrix']
shift = manifest['sampler_chart']['shift']
warp_strength = manifest['sampler_chart']['warp_strength']
warp_phase = manifest['sampler_chart']['warp_phase']
sigma_v = manifest['kinetic_metric']['sigma_v']
kinetic_metric = manifest['kinetic_metric']['inverse_mass_matrix']

# Read candidate transition bundle
candidate_ids = np.load('data/candidate_transition_bundle.npz')['candidate_ids']
step_size_ladder = np.load('data/candidate_transition_bundle.npz')['step_size_ladder']
probe_states = np.load('data/candidate_transition_bundle.npz')['probe_states']
probe_momenta = np.load('data/candidate_transition_bundle.npz')['probe_momenta']
proposal_states = np.load('data/candidate_transition_bundle.npz')['proposal_states']
proposal_momenta = np.load('data/candidate_transition_bundle.npz')['proposal_momenta']
reverse_states = np.load('data/candidate_transition_bundle.npz')['reverse_states']
reverse_momenta = np.load('data/candidate_transition_bundle.npz')['reverse_momenta']
volume_input = np.load('data/candidate_transition_bundle.npz')['volume_input']
volume_output = np.load('data/candidate_transition_bundle.npz')['volume_output']

# Read candidate chain panel
panel_centered_samples = np.load('data/candidate_chain_panel.npz')['panel_centered_samples']
panel_delta_h = np.load('data/candidate_chain_panel.npz')['panel_delta_h']
panel_accepted = np.load('data/candidate_chain_panel.npz')['panel_accepted']

# Read production replay streams
initial_latent_states = np.load('data/initial_latent_states.npy')
transition_momenta = np.load('data/transition_momenta.npy')
transition_uniforms = np.load('data/transition_uniforms.npy')

# Read held-out replay streams
holdout_initial_latent_states = np.load('data/holdout_initial_latent_states.npy')
holdout_transition_momenta = np.load('data/holdout_transition_momenta.npy')
holdout_transition_uniforms = np.load('data/holdout_transition_uniforms.npy')

# Read profile bins
profile_bins = pd.read_csv('data/profile_bins.csv')

# Candidate-kernel audit
audit = []
for j in range(len(candidate_ids)):
    candidate_id = candidate_ids[j]
    step_size = step_size_ladder[j]
    energy_rmse = np.sqrt(np.mean((H(proposal_states[j], proposal_momenta[j]) - H(probe_states[j], probe_momenta[j]))**2))
    energy_slope = np.polyfit(np.log(step_size_ladder), np.log(energy_rmse), 1)[0]
    reverse_p95 = np.percentile(np.sum(np.abs(reverse_states[j] - probe_states[j]) + np.abs(reverse_momenta[j] + probe_momenta[j]), axis=1), 95)
    volume_logdet