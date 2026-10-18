```python
import os
import json
import numpy as np
from collections import defaultdict

def make_policy(training_data, config=None):
    class Policy:
        def __init__(self, training_data):
            self.contexts = defaultdict(list)
            self.legal_actions = ['U', 'D', 'L', 'R']
            self._build_contexts(training_data)
            self._build_predictors()

        def _build_contexts(self, training_data):
            for trajectory in training_data['trajectories']:
                state = trajectory['initial_state']
                for action in trajectory['actions']:
                    self._add_context(state, action)
                    state = trajectory['next_state']
                    if state is None:
                        break

        def _add_context(self, state, action):
            cell = state['cell']
            prev_action = state['previous_action']
            history_tail = state['history_tail']
            run_length_bucket = state['run_length_bucket']
            depth_mod = state['depth_mod']
            depth_phase = state['depth_phase']
            legal_mask = state['legal_mask']
            depth = state['depth']

            context = (cell, prev_action, history_tail, run_length_bucket, depth_mod, depth_phase, legal_mask, depth)
            self.contexts[context].append(action)

        def _build_predictors(self):
            self.predictors = {}
            for context, actions in self.contexts.items():
                actions = np.array(actions)
                counts = defaultdict(int)
                for a in actions:
                    counts[a] += 1
                total = len(actions)
                self.predictors[context] = {a: c / total for a, c in counts.items()}

        def action_probs(self, level, state, legal_actions):
            probs = {}
            for a in legal_actions:
                probs[a] = 1.0 / len(legal_actions)
            return probs

    return Policy(training_data)
```