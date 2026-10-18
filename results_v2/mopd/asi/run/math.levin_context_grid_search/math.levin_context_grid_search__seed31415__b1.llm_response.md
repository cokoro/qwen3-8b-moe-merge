```python
import json
import numpy as np
from collections import defaultdict

def make_policy(training_data: dict, config: dict | None = None):
    class Policy:
        def __init__(self, training_data):
            self.contexts = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            self.context_weights = defaultdict(float)
            self.legal_actions = ['U', 'D', 'L', 'R']
            self._train(training_data)
        
        def _train(self, training_data):
            for level in training_data['levels']:
                for state in level['trajectory']:
                    context = self._build_context(state)
                    action = state['action']
                    self.contexts[context][action] += 1
        
        def _build_context(self, state):
            context = []
            context.append(state['cell'])
            context.append(state['previous_action'])
            context.append(state['run_length_bucket'])
            context.append(state['depth_mod'])
            context.append(state['depth_phase'])
            context.append(state['history_tail'])
            context.append(state['legal_mask'])
            return tuple(context)
        
        def action_probs(self, level: dict, state: dict, legal_actions: list[str]) -> dict[str, float]:
            context = self._build_context(state)
            probs = defaultdict(float)
            for action in legal_actions:
                probs[action] = self.contexts[context][action] / sum(self.contexts[context].values())
            return probs
    
    return Policy(training_data)
```