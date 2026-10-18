import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import scipy.optimize
import scipy.linalg

# Load input data
with open("data/target_manifest.json") as f:
    target_manifest = json.load(f)

with open("data/transition_schedule.json") as f:
    transition_schedule = json.load(f)

candidate_bundle = np.load("data/candidate_transition_bundle.npz")
candidate_ids = candidate_bundle["candidate_ids"]
step_size_ladder = candidate_bundle["step_size_ladder"]
probe_states = candidate_bundle["probe_states"]
probe_momenta = candidate_bundle["probe_momenta"]
proposal_states = candidate_bundle["proposal_states"]
proposal_momenta = candidate_bundle["proposal_momenta"]
reverse_states = candidate_bundle["reverse_states"]
reverse_momenta = candidate_bundle["reverse_momenta"]
volume_input = candidate_bundle["volume_input"]
volume_output = candidate_bundle["volume_output"]

chain_panel = np.load("data/candidate_chain_panel.npz")
panel_centered_samples = chain_panel["panel_centered_samples"]
panel_delta_h = chain_panel["panel_delta_h"]
panel_accepted = chain_panel["panel_accepted"]

initial_latent_states = np.load("data/initial_latent_states.npy")
transition_momenta = np.load("data/transition_momenta.npy")
transition_uniforms = np.load("data/transition_uniforms.npy")

holdout_initial_latent_states = np.load("data/holdout_initial_latent_states.npy")
holdout_transition_momenta = np`