import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from scipy.stats import norm
import os
import json

def load_data(case_dir):
    expr = pd.read_csv(f"{case_dir}/expression_matrix.csv", index_col=0)
    metadata = pd.read_csv(f"{case_dir}/cell_metadata.csv", index_col=0)
    spec = json.load(open(f"{case_dir}/analysis_spec.json"))
    return expr, metadata, spec

def run_analysis(case_dir):
    expr, metadata, spec = load_data(case_dir)
    conditions = spec['conditions']
    alpha = spec['alpha']
    K = spec.get('K', 5)
    
    control_cells = metadata[metadata.condition == conditions[0]].index
    a_cells = metadata[metadata.condition == conditions[1]].index
    b_cells = metadata[metadata.condition == conditions[2]].index
    expr = expr.loc[expr.index.intersection(control_cells.union(a_cells).union(b_cells))]
    
    expr = expr.values
    metadata = metadata.loc[expr.index]
    control_idx = np.where(metadata.condition == conditions[0])[0]
    a_idx = np.where(metadata.condition == conditions[1])[0]
    b_idx = np.where(metadata.condition == conditions[2])[0]
    
    def run_direction(Y, X, a_idx, b_idx, K, n_folds=K):
        n = len(X)
        fold_indices = KFold(n_splits=K).split(np.zeros(n))
        fold_stats = []
        fold_vars = []
        fold_projections = []
        
        for train_idx, test_idx in fold_indices:
            train_cells = np.concatenate([a_idx, b_idx])
            train_cells = train_cells[np.isin(train_cells, np.array(list(train_idx)))]
            test_cells = np.array(list(test_idx))
            train_cells = train_cells[np.isin(train_cells, control_idx)]
            train_cells = np.array(list(train_idx))
            train_cells = train_cells[np.isin(train_cells, control_idx)]
            train_cells = np.concatenate([train_cells, a_idx, b_idx])
            train_cells = train_cells[np.isin(train_cells, np.array(list(train_idx)))]
            test_cells = np.array(list(test_idx))
            test_cells = test_cells[np.isin(test_cells, control_idx)]
            test_cells = np.concatenate([test_cells, a_idx, b_idx])
            test_cells = test_cells[np.isin(test_cells, np.array(list(test_idx)))]
            test_cells = np.array(list(test_idx))
            
            train_X = expr[train_cells]
            train_Y = Y[train_cells]
            test_X = expr[test_cells]
            test_Y = Y[test_cells]
            
            beta1 = Lasso(alpha=0.1).fit(train_X, train_Y).coef_
            s1 = np.where(beta1 != 0)[0]
            train_X = train_X[:, s1]
            test_X = test_X[:, s1]
            
            beta2 = Lasso(alpha=0.1).fit(train_X, train_Y).coef_
            s2 = np.where(beta2 != 0)[0]
            beta2 = np.zeros(len(expr[0]))
            beta2[s1] = beta2[s1]
            beta2[s2] = beta2[s2]
            
            pca = PCA(n_components=1)
            v = pca.fit_transform(expr[control_idx, s1])
            v = v / np.linalg.norm(v)
            w_n = (len(control_idx) ** (1/3))
            beta_proj = v + w_n * beta2
            beta_proj = beta_proj / np.linalg.norm(beta_proj)
            
            score_test = test_X @ beta_proj
            score_control = score_test[test_Y == 0]
            score_perturbation = score_test[test_Y == 1]
            T = np.mean(score_control) - np.mean(score_perturbation)
            Var = np.var(score_control) / len(score_control) + np.var(score_perturbation) / len(score_perturbation)
            fold_stats.append(T)
            fold_vars.append(Var)
            fold_projections.append(beta_proj)
        
        baseline_proj = fold_projections[0]
        aligned_stats = []
        for i in range(1, len(fold_projections)):
            dot = np.dot(fold_projections[i], baseline_proj)
            if dot < 0:
                fold_stats[i] *= -1
            aligned_stats.append(fold_stats[i])
        
        T_bar = np.mean(aligned_stats)
        SE = np.sqrt(np.sum(fold_vars) / len(aligned_stats)**2)
        z = T_bar / SE
        p = 2 * norm.cdf(-abs(z))
        
        return p, T_bar, SE, aligned_stats
    
    p_a_to_b, T_a_to_b, SE_a_to_b, aligned_stats_a_to_b = run_direction(np.zeros(len(expr)), expr, a_idx, b_idx, K)
    p_b_to_a, T_b_to_a, SE_b_to_a, aligned_stats_b_to_a = run_direction(np.zeros(len(expr)), expr, b_idx, a_idx, K)
    
    shared_genes = np.zeros(len(expr[0]))
    for i in range(K):
        for j in range(len(aligned_stats_a_to_b[i])):
            if aligned_stats_a_to_b[i][j] > 0:
                shared_genes[j] = 1
    for i in range(K):
        for j in range(len(aligned_stats_b_to_a[i])):
            if aligned_stats_b_to_a[i][j] > 0:
                shared_genes[j] = 1
    
    p_combined = min(p_a_to_b, p_b_to_a)
    convergence_decision = 1 if p_combined < alpha else 0
    
    os.makedirs(f"results/{case_dir}", exist_ok=True)
    np.savetxt(f"results/{case_dir}/shared_feature_mask.npy", shared_genes.astype(int), fmt='%i')
    np.savetxt(f"results/{case_dir}/convergence_decision.npy", [convergence_decision], fmt='%i')
    np.savetxt(f"results/{case_dir}/convergence_pvalues.csv", [[p_a_to_b, p_b_to_a, p_combined], [T_a_to_b, T_b_to_a, 0], [SE_a_to_b, SE_b_to_a, 0]], delimiter=',', header='direction,p_value,z_statistic,n_selected_features')
    
    summary = {
        'convergent': convergence_decision,
        'alpha': alpha,
        'primary_p_value': p_combined,
        'a_to_b_p_value': p_a_to_b,
        'b_to_a_p_value': p_b_to_a,
        'n_selected_features': np.sum(shared_genes),
        'shared_gene_names': [f"G{idx:04d}" for idx in np.where(shared_genes == 1)[0]],
        'interpretation': 'Convergent' if convergence_decision else 'Not convergent'
    }
    with open(f"results/{case_dir}/convergence_summary.json", 'w') as f:
        json.dump(summary, f)
    
    plt.figure(figsize=(10, 6))
    sns.histplot([T_a_to_b, T_b_to_a], kde=True, stat='density', label=['A_to_B', 'B_to_A'])
    plt.title('Fold Statistics Distribution')
    plt.legend()
    plt.savefig(f"results/{case_dir}/convergence_diagnostic.png")
    plt.close()

run_analysis('case_0')
run_analysis('case_1')