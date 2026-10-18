```python
import os
import json
import numpy as np
from collections import defaultdict

class Policy:
    def __init__(self, model):
        self.model = model

    def action_probs(self, level, state, legal_actions):
        obs = self._extract_obs(state)
        if not self.model:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
        return self.model.predict(obs, legal_actions)

    def _extract_obs(self, state):
        obs = {
            'cell': state.get('cell', ''),
            'neighbors': state.get('neighbors', []),
            'previous_action': state.get('previous_action', ''),
            'history': state.get('history', []),
            'history_tail': state.get('history_tail', []),
            'run_length_bucket': state.get('run_length_bucket', 0),
            'depth': state.get('depth', 0),
            'depth_mod': state.get('depth_mod', 0),
            'depth_phase': state.get('depth_phase', 0),
            'legal_mask': state.get('legal_mask', []),
            'legal_actions': state.get('legal_actions', [])
        }
        return obs

def make_policy(training_data: dict, config: dict | None = None):
    model = ContextModel(training_data)
    return Policy(model)

class ContextModel:
    def __init__(self, training_data):
        self.contexts = defaultdict(lambda: defaultdict(lambda: np.zeros(4)))
        self.likelihoods = defaultdict(lambda: defaultdict(float))
        self._train(training_data)

    def _train(self, training_data):
        for traj in training_data['trajectories']:
            state = traj['initial_state']
            for step in traj['steps']:
                obs = self._extract_obs(state)
                action = step['action']
                self._update(obs, action)
                state = step['next_state']

    def _extract_obs(self, state):
        obs = {
            'cell': state.get('cell', ''),
            'neighbors': state.get('neighbors', []),
            'previous_action': state.get('previous_action', ''),
            'history': state.get('history', []),
            'history_tail': state.get('history_tail', []),
            'run_length_bucket': state.get('run_length_bucket', 0),
            'depth': state.get('depth', 0),
            'depth_mod': state.get('depth_mod', 0),
            'depth_phase': state.get('depth_phase', 0),
            'legal_mask': state.get('legal_mask', []),
            'legal_actions': state.get('legal_actions', [])
        }
        return obs

    def _update(self, obs, action):
        for context in self._generate_contexts(obs):
            self._update_context(context, action)

    def _generate_contexts(self, obs):
        contexts = []
        for k in obs:
            contexts.append(k)
        return contexts

    def _update_context(self, context, action):
        for key in self.contexts:
            self.contexts[key][context][action_index(action)] += 1

    def predict(self, obs, legal_actions):
        prob = defaultdict(float)
        for key in self.contexts:
            for a in legal_actions:
                prob[a] += self.contexts[key][obs][a_index(a)] * self.likelihoods[key]
        return {a: prob[a] for a in legal_actions}

def a_index(a):
    return {'U': 0, 'D': 1, 'L': 2, 'R': 3}[a]
```