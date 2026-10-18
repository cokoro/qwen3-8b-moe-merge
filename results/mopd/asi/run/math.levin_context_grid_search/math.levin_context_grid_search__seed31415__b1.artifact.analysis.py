import json
import numpy as np
from collections import defaultdict

class Policy:
    def __init__(self, training_data):
        self.contexts = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        self.context_weights = defaultdict(lambda: np.zeros(4))
        self.legal_actions = ['U', 'D', 'L', 'R']
        self.depth_mod = 8
        self.depth_phase = 4
        self.run_length_buckets = 4
        self.train(training_data)
        
    def train(self, training_data):
        for level in training_data['levels']:
            for state in level['trajectory']:
                context = self._make_context(state)
                action_idx = self.legal_actions.index(state['action'])
                self.contexts[context][action_idx] += 1
                
    def _make_context(self, state):
        cell = state['cell']
        prev_action = state['previous_action']
        history_tail = state['history_tail']
        run_length_bucket = state['run_length_bucket']
        depth_mod = state['depth'] % self.depth_mod
        depth_phase = state['depth'] // self.depth_phase
        return f"{cell}|{prev_action}|{history_tail}|{run_length_bucket}|{depth_mod}|{depth_phase}"
    
    def action_probs(self, level, state, legal_actions):
        context = self._make_context(state)
        action_probs = np.zeros(4)
        for i, action in enumerate(legal_actions):
            action_probs[i] = self.contexts[context][i] + 1
        action_probs /= np.sum(action_probs)
        return {action: prob for action, prob in zip(legal_actions, action_probs)}