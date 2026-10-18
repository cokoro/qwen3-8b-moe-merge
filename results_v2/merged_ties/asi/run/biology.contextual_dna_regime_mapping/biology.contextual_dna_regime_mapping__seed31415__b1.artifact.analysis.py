```bash
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.optimize import Bounds

from ..utils import load_sequence_bundle
from ..utils import load_analysis_spec
from ..utils import write_predicted_regions
from ..utils import write_state_paths
from ..utils import write_posterior_probs
from ..utils import write_fit_diagnostics
from ..utils import write_summary_metrics

from ..plotting import plot_sequence_posterior

from ..base import _get_dinucleotide_order

def run_analysis(spec_path='data/analysis_spec.json'):
    # Load analysis spec
    spec = load_analysis_spec(spec_path)

    # Load sequences
    sequences = load_sequence_bundle(spec['sequence_file'])

    # Compute adjacent-pair order
    order = _get_dinucleotide_order(spec['adjacent_pair_order'])

    # Build HMM
    hmm = HiddenMarkovModel(spec['n_states'], order=order)
    hmm.init_from_data(sequences)

    # Run EM
    for iter_num in range(spec['n_em_iterations']):
        log_likelihood = hmm.estimate(sequences)

    # Compute predicted regions
    state_paths = hmm.viterbi(sequences)
    posterior_probs = hmm.posterior(sequences)

    # Write outputs
    write_predicted_regions(spec['predicted_regions_file'], state_paths, spec['window_size'])
    write_state_paths(spec['state_paths_file'], state_paths)
    write_posterior_probs(spec['posterior_probs_file'], posterior_probs)
    write_fit_diagnostics(spec['fit_diagnostics_file'], hmm)
    write_summary_metrics(spec['summary_metrics_file'], state_paths)
    plot_sequence_posterior(spec['diagnostic_plots_file'], state_paths)

    return state_paths, posterior_probs

class HiddenMarkovModel:
    def __init__(self, n_states, order=16, pseudocount=1.0):
        self.n_states = n_states
        self.order = order
        self.pseudocount = pseudocount
        self.log_pi = np.zeros(n_states)
        self.log_A = np.zeros((n_states, n_states))
        self.log_B = np.zeros((n_states, order))

    def init_from_data(self, sequences):
        # Initialize transition matrix
        self.log_A = np.log(np.ones((self.n_states, self.n_states)) * self.pseudocount)

        # Initialize emission matrix
        self.log_B = np.log(np.ones((self.n_states, self.order)) * self.pseudocount)

        # Initialize prior
        self.log_pi = np.log(np.ones(self.n_states) * self.pseudocount)

        for seq in sequences:
            for o in seq['observations']:
                for j in range(self.n_states):
                    self.log_B[j, o] += self.pseudocount

        for j in range(self.n_states):
            self.log_B[j] /= self.pseudocount * self.order

        for j in range(self.n_states):
            self.log_pi[j] /= self.pseudocount

    def estimate(self, sequences):
        for seq in sequences:
            for t in range(len(seq['observations'])):
                for j in range(self.n_states):
                    for i in range(self.n_states):
                        self.log_A[i, j] += self.log_B[j, seq['observations'][t]]
                        self.log_A[i, j] += self.log_A[i, j]

        return log_likelihood

    def viterbi(self, sequences):
        for seq in sequences:
            for t in range(len(seq['observations'])):
                for j in range(self.n_states):
                    for i in range(self.n_states):
                        self.log_A[i, j] += self.log_B[j, seq['observations'][t]]
                        self.log_A[i, j] += self.log_A[i, j]

        return log_likelihood

    def posterior(self, sequences):
        for seq in sequences:
            for t in range(len(seq['observations'])):
                for j in range(self.n_states):
                    for i in range(self.n_states):
                        self.log_A[i, j] += self.log_B[j, seq['observations'][t]]
                        self.log_A[i, j] += self.log_A[i, j]

        return log_likelihood

```