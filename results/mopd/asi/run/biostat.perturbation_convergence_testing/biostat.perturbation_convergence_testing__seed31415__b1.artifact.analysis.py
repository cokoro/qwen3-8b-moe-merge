import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import norm
import os
import json

def load_data(case_dir):
    expr = pd.read_csv(f"{case_dir}/expression_matrix.csv", index_col=0)
    meta = pd.read_csv(f"{case_dir}/cell_metadata.csv")
    spec = json.load(open(f"{case_dir}/analysis_spec.json"))
    return expr, meta, spec

def run_analysis(case_dir):
    expr, meta, spec = load_data(case_dir)
    conditions = spec["condition_labels"]
    control = meta[meta["condition"] == conditions[0]]
    perturbation_a = meta[meta["condition"] == conditions[1]]
    perturbation_b = meta[meta["condition"] == conditions[2]]
    
    expr_control = expr.loc[control["cell_id"], :].values
    expr_a = expr.loc[perturbation_a["cell_id"], :].values
    expr_b = expr.loc[perturbation_b["cell_id"], :].values
    
    K = 5
    kf = KFold(n_splits=K)
    
    def run_direction_a_to_b(expr_a, expr_b, expr_control):
        fold_stats = []
        fold_vars = []
        for train_idx, val_idx in kf.split(expr_control):
            expr_train_control = expr_control[train_idx]
            expr_train_a = expr_a[train_idx]
            expr_train_b = expr_b[train_idx]
            expr_val_control = expr_control[val_idx]
            expr_val_a = expr_a[val_idx]
            expr_val_b = expr_b[val_idx]
            
            # Step 3: Direction A_to_B
            # Step 3.1: Select candidate features from A vs control
            X = np.vstack([expr_train_control, expr_train_a])
            Y = np.array([0]*len(expr_train_control) + [1]*len(expr_train_a))
            model = Lasso(alpha=0.1)
            model.fit(X, Y)
            s1 = np.where(model.coef_ != 0)[0]
            
            # Step 3.2: Select candidate features from B vs control within s1
            X = np.vstack([expr_train_control[:, s1], expr_train_b[:, s1]])
            Y = np.array([0]*len(expr_train_control) + [1]*len(expr_train_b))
            model = Lasso(alpha=0.1)
            model.fit(X, Y)
            s2 = np.where(model.coef_ != 0)[0]
            s2_full = np.zeros(len(expr_control.columns), dtype=int)
            s2_full[s1[s2]] = 1
            
            # Step 3.3: Compute projection direction
            X = expr_train_control[:, s1]
            pca = PCA(n_components=1)
            v_hat = pca.fit_transform(X)
            v_hat = np.zeros(len(expr_control.columns))
            v_hat[s1] = v_hat[0]
            beta2_hat = model.coef_
            beta_proj = v_hat + (len(expr_train_control) ** (1/3)) * beta2_hat
            v = beta_proj / np.linalg.norm(beta_proj)
            
            # Step 4: Compute scores on validation data
            score_control = expr_val_control @ v
            score_a = expr_val_a @ v
            
            # Step 5: Compute fold statistic
            T = np.mean(score_control) - np.mean(score_a)
            Var = np.var(score_control) / len(score_control) + np.var(score_a) / len(score_a)
            
            # Step 7: Align projection signs
            if len(fold_stats) == 0:
                v_baseline = v
            else:
                dot_product = v @ v_baseline
                if dot_product < 0:
                    T = -T
            fold_stats.append(T)
            fold_vars.append(Var)
        
        # Step 8: Aggregate
        T_bar = np.mean(fold_stats)
        SE = np.sqrt(np.sum(fold_vars) / K**2)
        z = T_bar / SE
        p_value = 2 * norm.cdf(-abs(z))
        return p_value, fold_stats, fold_vars
    
    p_a_to_b, fold_stats_a, fold_vars_a = run_direction_a_to_b(expr_a, expr_b, expr_control)
    p_b_to_a, fold_stats_b, fold_vars_b = run_direction_b_to_a(expr_b, expr_a, expr_control)
    
    # Step 9: Combine directional results
    primary_p_value = min(p_a_to_b, p_b_to_a)
    
    # Step 10: Create shared feature mask
    shared_mask = np.zeros(len(expr.columns), dtype=int)
    for fold in range(K):
        # Implement logic to compute shared genes
        # This is a placeholder for the actual implementation
        pass
    
    # Step 11: Create convergence decision
    convergent = 1 if primary_p_value < spec["alpha"] else 0
    
    # Step 12: Save outputs
    os.makedirs(f"results/{case_dir}", exist_ok=True)
    np.save(f"results/{case_dir}/convergence_decision.npy", convergent)
    np.save(f"results/{case_dir}/shared_feature_mask.npy", shared_mask)
    pd.DataFrame({
        "direction": ["a_to_b", "b_to_a", "combined"],
        "p_value": [p_a_to_b, p_b_to_a, primary_p_value],
        "z_statistic": [T_bar, T_bar, z],
        "n_selected_features": [len(s1), len(s2), len(shared_mask)]
    }).to_csv(f"results/{case_dir}/convergence_pvalues.csv", index=False)
    convergence_summary = {
        "convergent": convergent,
        "alpha": spec["alpha"],
        "primary_p_value": primary_p_value,
        "p_a_to_b": p_a_to_b,
        "p_b_to_a": p_b_to_a,
        "n_selected_features": len(s1),
        "shared_gene_names": [f"G{idx:04d}" for idx in np.where(shared_mask == 1)[0]],
        "interpretation": "Convergent perturbations have shared effects."
    }
    with open(f"results/{case_dir}/convergence_summary.json", "w") as f:
        json.dump(convergence_summary, f)
    plt.figure()
    plt.hist(fold_stats_a, alpha=0.5, label="a_to_b")
    plt.hist(fold_stats_b, alpha=0.5, label="b_to_a")
    plt.legend()
    plt.savefig(f"results/{case_dir}/convergence_diagnostic.png")