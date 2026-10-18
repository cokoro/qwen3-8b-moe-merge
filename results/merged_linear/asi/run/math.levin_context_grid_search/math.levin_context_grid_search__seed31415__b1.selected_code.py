import json
import numpy as np
import math

def make_policy(training_data, config=None):
    return LocalContextPolicy(training_data)

class LocalContextPolicy:
    def __init__(self, training_data):
        self.training_data = training_data
        self.contexts = self.build_contexts()
        self.context_models = self.build_context_models()
        self.context_weights = self.train_context_weights()
    
    def build_contexts(self):
        contexts = {
            "cell": set(),
            "previous_action": set(),
            "run_length_bucket": set(),
            "depth_mod": set(),
            "depth_phase": set(),
            "legal_mask": set(),
            "history_tail": set(),
            "cell_previous_action_run_length_depth_mod": set(),
            "cell_history_tail_depth_mod": set(),
            "cell_previous_action_depth_mod": set(),
            "cell_run_length_depth_mod": set(),
            "cell_depth_mod": set()
        }
        
        for level in self.training_data["levels"]:
            for state in level["trajectory"]:
                cell = state["cell"]
                previous_action = state["previous_action"]
                run_length_bucket = state["run_length_bucket"]
                depth_mod = state["depth_mod"]
                depth_phase = state["depth_phase"]
                legal_mask = state["legal_mask"]
                history_tail = state["history_tail"]
                
                contexts["cell"].add(cell)
                contexts["previous_action"].add(previous_action)
                contexts["run_length_bucket"].add(run_length_bucket)
                contexts["depth_mod"].add(depth_mod)
                contexts["depth_phase"].add(depth_phase)
                contexts["legal_mask"].add(legal_mask)
                contexts["history_tail"].add(history_tail)
                
                contexts["cell_previous_action_run_length_depth_mod"].add((cell, previous_action, run_length_bucket, depth_mod))
                contexts["cell_history_tail_depth_mod"].add((cell, history_tail, depth_mod))
                contexts["cell_previous_action_depth_mod"].add((cell, previous_action, depth_mod))
                contexts["cell_run_length_depth_mod"].add((cell, run_length_bucket, depth_mod))
                contexts["cell_depth_mod"].add((cell, depth_mod))
        
        return contexts
    
    def build_context_models(self):
        context_models = {}
        for context in self.contexts:
            model = {}
            for context_value in self.contexts[context]:
                model[context_value] = {}
                for action in ["U", "D", "L", "R"]:
                    model[context_value][action] = 1.0 / len(self.contexts[context])
            context_models[context] = model
        return context_models
    
    def train_context_weights(self):
        weights = {}
        for context in self.contexts:
            weights[context] = 1.0
        return weights
    
    def update_context_model(self, context, context_value, action, weight):
        model = self.context_models[context]
        model[context_value][action] += weight
    
    def estimate_log_prob(self, level, state, legal_actions):
        log_prob = 0.0
        for context in self.contexts:
            context_value = self.get_context_value(level, state, context)
            model = self.context_models[context]
            if context_value in model:
                action_probs = model[context_value]
                log_prob += np.log(action_probs[state["action"]])
        return log_prob
    
    def get_context_value(self, level, state, context):
        if context == "cell":
            return state["cell"]
        elif context == "previous_action":
            return state["previous_action"]
        elif context == "run_length_bucket":
            return state["run_length_bucket"]
        elif context == "depth_mod":
            return state["depth_mod"]
        elif context == "depth_phase":
            return state["depth_phase"]
        elif context == "legal_mask":
            return state["legal_mask"]
        elif context == "history_tail":
            return state["history_tail"]
        elif context == "cell_previous_action_run_length_depth_mod":
            return (state["cell"], state["previous_action"], state["run_length_bucket"], state["depth_mod"])
        elif context == "cell_history_tail_depth_mod":
            return (state["cell"], state["history_tail"], state["depth_mod"])
        elif context == "cell_previous_action_depth_mod":
            return (state["cell"], state["previous_action"], state["depth_mod"])
        elif context == "cell_run_length_depth_mod":
            return (state["cell"], state["run_length_bucket"], state["depth_mod"])
        elif context == "cell_depth_mod":
            return (state["cell"], state["depth_mod"])
        else:
            return None
    
    def action_probs(self, level, state, legal_actions):
        log_probs = {}
        for action in legal_actions:
            log_probs[action] = self.estimate_log_prob(level, state, legal_actions)
        return {action: np.exp(log_probs[action]) for action in legal_actions}