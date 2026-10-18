```python
import json
import numpy as np
from collections import defaultdict

def make_policy(training_data: dict, config: dict | None = None):
    class Policy:
        def __init__(self, training_data):
            self.contexts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            self.legal_actions = ['U', 'D', 'L', 'R']
            self._train(training_data)
        
        def _train(self, training_data):
            for level in training_data:
                for trajectory in level['trajectories']:
                    state = trajectory[0]
                    for action, next_state in zip(trajectory[1:], trajectory[2:]):
                        self._update_contexts(state, action)
                        state = next_state
        
        def _update_contexts(self, state, action):
            cell = state['cell']
            prev_action = state['previous_action'] if 'previous_action' in state else ''
            history_tail = state['history_tail'] if 'history_tail' in state else ''
            run_length_bucket = state['run_length_bucket'] if 'run_length_bucket' in state else 0
            depth_mod = state['depth_mod'] if 'depth_mod' in state else 0
            depth_phase = state['depth_phase'] if 'depth_phase' in state else 0
            legal_mask = state['legal_mask'] if 'legal_mask' in state else [1.0 for _ in self.legal_actions]
            
            for context in self._generate_contexts(cell, prev_action, history_tail, run_length_bucket, depth_mod, depth_phase):
                self.contexts[context][action] += 1
        
        def _generate_contexts(self, cell, prev_action, history_tail, run_length_bucket, depth_mod, depth_phase):
            contexts = []
            contexts.append(f"{cell}|{prev_action}|{run_length_bucket}|{depth_mod}|{depth_phase}")
            contexts.append(f"{cell}|{history_tail}|{depth_mod}|{depth_phase}")
            contexts.append(f"{cell}|{prev_action}|{depth_mod}")
            contexts.append(f"{cell}|{run_length_bucket}|{depth_mod}")
            contexts.append(f"{cell}|{depth_phase}")
            contexts.append(f"{cell}|{prev_action}|{history_tail}")
            contexts.append(f"{cell}|{run_length_bucket}|{depth_phase}")
            contexts.append(f"{cell}|{prev_action}|{depth_mod}")
            contexts.append(f"{cell}|{history_tail}|{depth_phase}")
            contexts.append(f"{cell}|{run_length_bucket}|{depth_mod}")
            return contexts
        
        def action_probs(self, level, state, legal_actions):
            probs = defaultdict(float)
            for action in legal_actions:
                prob = 1.0
                for context in self._generate_contexts(state['cell'], state['previous_action'], state['history_tail'], 
                                                      state['run_length_bucket'], state['depth_mod'], state['depth_phase']):
                    if context in self.contexts and action in self.contexts[context]:
                        prob *= (self.contexts[context][action] + 1) / (sum(self.contexts[context].values()) + 4)
                probs[action] = prob
            total = sum(probs.values())
            for action in probs:
                probs[action] /= total
            return probs
    
    return Policy(training_data)
```