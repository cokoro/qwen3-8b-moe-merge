import json
import numpy as np
from collections import defaultdict

def make_policy(training_data: dict, config: dict | None = None):
    class Policy:
        def __init__(self, training_data):
            self.contexts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            self.legal_actions = ['U', 'D', 'L', 'R']
            self._fit(training_data)
        
        def _fit(self, training_data):
            for traj in training_data['trajectories']:
                state = traj['state']
                for step in range(len(traj['actions'])):
                    current_cell = state['cell']
                    prev_action = traj['actions'][step-1] if step > 0 else None
                    run_len = traj['run_lengths'][step] if step < len(traj['run_lengths']) else 0
                    depth = traj['depth'][step]
                    depth_mod = depth % 4
                    depth_phase = depth // 4 % 2
                    history_tail = traj['history_tail'][step] if step < len(traj['history_tail']) else ''
                    legal_mask = traj['legal_mask'][step]
                    action = traj['actions'][step]
                    for c in self.contexts:
                        for k, v in self.contexts[c].items():
                            self.contexts[c][k][action] += v
                    for c in self.contexts:
                        for k, v in self.contexts[c].items():
                            self.contexts[c][k][action] /= sum(self.contexts[c][k].values())
        
        def action_probs(self, level, state, legal_actions):
            probs = defaultdict(float)
            for c in self.contexts:
                for k, v in self.contexts[c].items():
                    for a in legal_actions:
                        probs[a] += v[a] * (1.0 / len(legal_actions))
            return probs

    return Policy(training_data)