import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

def load_data(case_dir):
    expr_path = os.path.join(case_dir, 'expression_matrix.csv')
    meta_path = os.path.join(case_dir, 'cell_metadata.csv')
    spec_path = os.path.join(case_dir, 'analysis_spec.json')
    
    expr_df = pd.read_csv(expr_path, index_col=0)
    meta_df = pd.read_csv(meta_path, index_col=0)
    spec = json.load(open(spec_path))
    
    conditions = spec['conditions']
    control = conditions['control']
    perturbation_a = conditions['perturbation_a']
    perturbation_b = conditions['perturbation_b']
    
    expr_df = expr_df.join(meta_df[['condition']])
    expr_df = expr_df.set_index('condition')
    
    return expr_df, control, perturbation_a, perturbation_b, spec

def split_data(expr_df, control, perturbation_a, perturbation_b, K=5):
    expr_df = expr_df.reset_index()
    expr_df = expr_df.rename(columns={'index': 'condition'})
    
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    folds = []
    for train_idx, val_idx in kf.split(expr_df):
        train_df = expr_df.iloc[train_idx]
        val_df = expr_df.iloc[val_idx]
        folds.append((train_df, val_df))
    return folds

def fit_logistic_model(X, y, features, alpha=0.01):
    model = LogisticRegression(penalty='l1', solver='liblinear', C=1/alpha, max_iter=1000)
    model.fit(X[features], y)
    return model

def get_candidate_genes(model, features, threshold=0.01):
    coefs = model.coef_[0]
    selected = np.abs(coefs) > threshold
    return features[selected]

def compute_projection_direction(control_data, s1, beta2_hat, n_effective):
    control_data = control_data[s1]
    pca = PCA(n_components=1)
    v_hat = pca.fit_transform(control_data)
    v_hat = np.array(v_hat).flatten()
    v_hat = np.pad(v_hat, (0, len(s1) - len(v_hat)), 'constant')
    
    w_n = n_effective ** (1/3)
    beta_proj = v_hat + w_n * beta2_hat
    norm_beta_proj = np.linalg.norm(beta_proj)
    v = beta_proj / norm_beta_proj if norm_beta_proj > 0 else np.zeros_like(beta_proj)
    return v

def compute_fold_statistic(scores_control, scores_perturbation):
    mean_control = np.mean(scores_control)
    mean_perturbation = np.mean(scores_perturbation)
    T_j = mean_control - mean_perturbation
    var_control = np.var(scores_control) / len(scores_control)
    var_perturbation = np.var(scores_perturbation) / len(scores_perturbation)
    Var_j = var_control + var_perturbation
    return T_j, Var_j

def align_projection_directions(v_fold, v_baseline):
    dot_product = np.dot(v_fold, v_baseline)
    if dot_product < 0:
        return -v_fold
    return v_fold

def run_analysis(expr_df, control, perturbation_a, perturbation_b, spec, K=5):
    expr_df = expr_df.reset_index()
    expr_df = expr_df.rename(columns={'index': 'condition'})
    
    folds = split_data(expr_df, control, perturbation_a, perturbation_b, K)
    
    results = []
    shared_genes = set()
    
    for fold_idx, (train_df, val_df) in enumerate(folds):
        train_control = train_df[train_df['condition'] == control]
        train_a = train_df[train_df['condition'] == perturbation_a]
        train_b = train_df[train_df['condition'] == perturbation_b]
        
        features = train_control.columns[1:]
        y_a = (train_df['condition'] == perturbation_a).astype(int)
        model_a = fit_logistic_model(train_df[features], y_a, features)
        s1 = get_candidate_genes(model_a, features)
        
        y_b = (train_df['condition'] == perturbation_b).astype(int)
        model_b = fit_logistic_model(train_df[s1], y_b, s1)
        s2 = get_candidate_genes(model_b, s1)
        
        n_effective = min(len(train_control), len(train_b))
        v = compute_projection_direction(train_control[features], s1, model_b.coef_[0], n_effective)
        
        val_control = val_df[val_df['condition'] == control]
        val_b = val_df[val_df['condition'] == perturbation_b]
        
        scores_control = val_control[features].dot(v)
        scores_b = val_b[features].dot(v)
        
        T_j, Var_j = compute_fold_statistic(scores_control, scores_b)
        
        if fold_idx == 0:
            v_baseline = v
        else:
            v = align_projection_directions(v, v_baseline)
        
        results.append((T_j, Var_j))
    
    T_bar = np.mean([t for t, _ in results])
    Var_total = np.sum([var for _, var in results])
    SE = np.sqrt(Var_total / len(results)**2)
    z = T_bar / SE
    p_value = 2 * norm.cdf(-abs(z))
    
    return p_value, T_bar, SE, results, shared_genes

def main(case_dir):
    expr_df, control, perturbation_a, perturbation_b, spec = load_data(case_dir)
    p_value_a_to_b, T_bar_a_to_b, SE_a_to_b, results_a_to_b, shared_genes_a_to_b = run_analysis(expr_df, control, perturbation_a, perturbation_b, spec)
    
    expr_df_rev = expr_df.copy()
    expr_df_rev['condition'] = expr_df_rev['condition'].replace({perturbation_a: perturbation_b, perturbation_b: perturbation_a})
    
    p_value_b_to_a, T_bar_b_to_a, SE_b_to_a, results_b_to_a, shared_genes_b_to_a = run_analysis(expr_df_rev, control, perturbation_b, perturbation_a, spec)
    
    shared_genes = shared_genes_a_to_b.union(shared_genes_b_to_a)
    
    primary_p_value = min(p_value_a_to_b, p_value_b_to_a)
    
    results = {
        'convergent': 1 if primary_p_value < spec['alpha'] else 0,
        'alpha': spec['alpha'],
        'primary_p_value': primary_p_value,
        'p_a_to_b': p_value_a_to_b,
        'p_b_to_a': p_value_b_to_a,
        'n_selected_features': len(shared_genes),
        'shared_gene_names': list(shared_genes),
        'interpretation': 'Convergent' if primary_p_value < spec['alpha'] else 'Not convergent'
    }
    
    results_path = os.path.join('results', case_dir.split('/')[-1], 'convergence_summary.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f)
    
    pvalues_path = os.path.join('results', case_dir.split('/')[-1], 'convergence_pvalues.csv')
    os.makedirs(os.path.dirname(pvalues_path), exist_ok=True)
    df_pvalues = pd.DataFrame({
        'direction': ['a_to_b', 'b_to_a', 'combined'],
        'p_value': [p_value_a_to_b, p_value_b_to_a, primary_p_value],
        'z_statistic': [T_bar_a_to_b, T_bar_b_to_a, T_bar_a_to_b],
        'n_selected_features': [len(shared_genes_a_to_b), len(shared_genes_b_to_a), len(shared_genes)]
    })
    df_pvalues.to_csv(pvalues_path, index=False)
    
    mask = np.zeros(1941, dtype=np.int64)
    for gene in shared_genes:
        idx = int(gene[1:])
        mask[idx] = 1
    
    mask_path = os.path.join('results', case_dir.split('/')[-1], 'shared_feature_mask.npy')
    np.save(mask_path, mask)
    
    decision_path = os.path.join('results', case_dir.split('/')[-1], 'convergence_decision.npy')
    np.save(decision_path, results['convergent'])
    
    plt.figure(figsize=(10, 6))
    plt.hist([results_a_to_b[0][0], results_b_to_a[0][0]], bins=20, alpha=0.5, label=['a_to_b', 'b_to_a'])
    plt.xlabel('Fold Statistic')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Fold Statistics for a_to_b and b_to_a')
    plt.savefig(os.path.join('results', case_dir.split('/')[-1], 'convergence_diagnostic.png'))
    plt.close()

if __name__ == '__main__':
    main('data/case_0')
    main('data/case_1')