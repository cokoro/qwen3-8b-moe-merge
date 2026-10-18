import json
import numpy as np
from collections import defaultdict, Counter

def make_policy(training_data: dict, config: dict | None = None):
    class Policy:
        def __init__(self, training_data):
            self.contexts = defaultdict(lambda: defaultdict(Counter))
            self.contexts['cell'] = defaultdict(Counter)
            self.contexts['prev_act'] = defaultdict(Counter)
            self.contexts['history_tail'] = defaultdict(Counter)
            self.contexts['depth_mod'] = defaultdict(Counter)
            self.contexts['run_length'] = defaultdict(Counter)
            self.contexts['depth_phase'] = defaultdict(Counter)
            self.contexts['cell_prev_act_run_length_depth_mod'] = defaultdict(Counter)
            self.contexts['cell_history_tail_depth_mod'] = defaultdict(Counter)
            self.legal_actions = ['U', 'D', 'L', 'R']
            self._train(training_data)

        def _train(self, training_data):
            for level in training_data['levels']:
                for traj in level['trajectories']:
                    state = traj[0]
                    for action, next_state in zip(traj[1:], traj[2:]):
                        self._update_contexts(state, action, next_state)
                        state = next_state

        def _update_contexts(self, state, action, next_state):
            for ctx in self.contexts:
                if ctx == 'cell':
                    self.contexts[ctx][state['cell']][action] += 1
                elif ctx == 'prev_act':
                    self.contexts[ctx][state['previous_action']][action] += 1
                elif ctx == 'history_tail':
                    self.contexts[ctx][tuple(state['history_tail'])][action] += 1
                elif ctx == 'depth_mod':
                    self.contexts[ctx][state['depth_mod']][action] += 1
                elif ctx == 'run_length':
                    self.contexts[ctx][state['run_length_bucket']][action] += 1
                elif ctx == 'depth_phase':
                    self.contexts[ctx][state['depth_phase']][action] += 1
                elif ctx == 'cell_prev_act_run_length_depth_mod':
                    key = (state['cell'], state['previous_action'], state['run_length_bucket'], state['depth_mod'])
                    self.contexts[ctx][key][action] += 1
                elif ctx == 'cell_history_tail_depth_mod':
                    key = (state['cell'], tuple(state['history_tail']), state['depth_mod'])
                    self.contexts[ctx][key][action] += 1

        def action_probs(self, level: dict, state: dict, legal_actions: list[str]) -> dict[str, float]:
            probs = {}
            for action in legal_actions:
                prob = 1.0
                for ctx in self.contexts:
                    if ctx == 'cell':
                        if state['cell'] in self.contexts[ctx]:
                            prob *= self.contexts[ctx][state['cell']].get(action, 0) / len(self.contexts[ctx][state['cell']])
                    elif ctx == 'prev_act':
                        if state['previous_action'] in self.contexts[ctx]:
                            prob *= self.contexts[ctx][state['previous_action']].get(action, 0) / len(self.contexts[ctx][state['previous_action']])
                    elif ctx == 'history_tail':
                        if tuple(state['history_tail']) in self.contexts[ctx]:
                            prob *= self.contexts[ctx][tuple(state['history_tail'])].get(action, 0) / len(self.contexts[ctx][tuple(state['history_tail'])])
                    elif ctx == 'depth_mod':
                        if state['depth_mod'] in self.contexts[ctx]:
                            prob *= self.contexts[ctx][state['depth_mod']].get(action, 0) / len(self.contexts[ctx][state['depth_mod']])
                    elif ctx == 'run_length':
                        if state['run_length_bucket'] in self.contexts[ctx]:
                            prob *= self.contexts[ctx][state['run_length_bucket']].get(action, 0) / len(self.contexts[ctx][state['run_length_bucket']])
                    elif ctx == 'depth_phase':
                        if state['depth_phase'] in self.contexts[ctx]:
                            prob *= self.contexts[ctx][state['depth_phase']].get(action, 0) / len(self.contexts[ctx][state['depth_phase']])
                    elif ctx == 'cell_prev_act_run_length_depth_mod':
                        key = (state['cell'], state['previous_action'], state['run_length_bucket'], state['depth_mod'])
                        if key in self.contexts[ctx]:
                            prob *= self.contexts[ctx][key].get(action, 0) / len(self.contexts[ctx][key])
                    elif ctx == 'cell_history_tail_depth_mod':
                        key = (state['cell'], tuple(state['history_tail']), state['depth_mod'])
                        if key in self.contexts[ctx]:
                            prob *= self.contexts[ctx][key].get(action, 0) / len(self.contexts[ctx][key])
                probs[action] = prob
            return probs

    policy = Policy(training_data)
    return policy