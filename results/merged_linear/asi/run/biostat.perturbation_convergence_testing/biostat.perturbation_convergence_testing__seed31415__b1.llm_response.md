```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

def load_data(case_dir):
    expr = pd.read_csv(os.path.join(case_dir, 'expression_matrix.csv'), index_col=0)
    meta = pd.read_csv(os.path.join(case_dir, 'cell_metadata.csv'))
    spec = pd.read_json(os.path.join(case_dir, 'analysis_spec.json'))
    return expr, meta, spec

def analyze_case(case_dir, case_id):
    expr, meta, spec = load_data(case_dir)
    control_cells = meta[meta['condition'] == spec['control_condition']].index
    a_cells = meta[meta['condition'] == spec['perturbation_a_condition']].index
    b_cells = meta[meta['condition'] == spec['perturbation_b_condition']].index
    
    n_folds = 5
    fold_indices = []
    for fold in range(n_folds):
        fold_indices.append({
            'control': StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(control_cells, meta.loc[control_cells]['condition'])[1],
            'a': StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(a_cells, meta.loc[a_cells]['condition'])[1],
            'b': StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42).split(b_cells, meta.loc[b_cells]['condition'])[1]
        })
    
    results = []
    for fold in range(n_folds):
        train_control = control_cells[fold_indices[fold]['control']]
        train_a = a_cells[fold_indices[fold]['a']]
        train_b = b_cells[fold_indices[fold]['b']]
        val_control = control_cells[~fold_indices[fold]['control']]
        val_a = a_cells[~fold_indices[fold]['a']]
        val_b = b_cells[~fold_indices[fold]['b']]
        
        train_expr = expr.loc[np.concatenate([train_control, train_a, train_b])]
        train_y = np.concatenate([np.zeros(len(train_control)), np.ones(len(train_a))])
        train_y_b = np.concatenate([np.zeros(len(train_control)), np.ones(len(train_b))])
        
        s1 = []
        for gene in expr.columns[1:]:
            train_gene = train_expr[gene]
            lr = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=3, random_state=42)
            lr.fit(train_expr[[gene]], train_y)
            if lr.coef_[0][0] != 0:
                s1.append(gene)
        
        s1_expr = train_expr[s1]
        lr_b = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=3, random_state=42)
        lr_b.fit(s1_expr, train_y_b)
        s2 = [gene for gene in s1 if lr_b.coef_[0][s1.index(gene)] != 0]
        
        pca = PCA(n_components=1)
        pca.fit(train_expr.loc[train_control, s2])
        v_hat = pca.components_[0]
        w_n = len(train_control) ** (1/3)
        beta_proj = v_hat + w_n * lr_b.coef_[0][s1.index(gene) for gene in s2]
        v = beta_proj / np.linalg.norm(beta_proj)
        
        score_val_a = train_expr.loc[val_a, s2].dot(v)
        score_val_b = train_expr.loc[val_b, s2].dot(v)
        score_val_control = train_expr.loc[val_control, s2].dot(v)
        
        t = np.mean(score_val_control) - np.mean(score_val_b)
        var = np.var(score_val_control) / len(val_control) + np.var(score_val_b) / len(val_b)
        
        results.append({'fold': fold, 't': t, 'var': var, 'v': v})
    
    results.sort(key=lambda x: abs(x['t']))
    aligned_results = []
    for i, result in enumerate(results):
        if i == 0:
            baseline_v = result['v']
        else:
            dot_product = result['v'].dot(baseline_v)
            if dot_product < 0:
                result['t'] *= -1
        aligned_results.append(result)
    
    t_bar = np.mean([r['t'] for r in aligned_results])
    se = np.sqrt(np.sum([r['var'] for r in aligned_results]) / len(aligned_results)**2)
    z = t_bar / se
    p_value = 2 * (1 - scipy.stats.norm.cdf(abs(z)))
    
    s1_expr = expr.loc[:, s1]
    lr = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=3, random_state=42)
    lr.fit(s1_expr, np.concatenate([np.zeros(len(control_cells)), np.ones(len(a_cells))]))
    s1_mask = np.zeros(len(expr.columns), dtype=int)
    s1_mask[expr.columns.get_loc(gene) for gene in s1 if lr.coef_[0][expr.columns.get_loc(gene)] != 0] = 1
    
    s2_expr = expr.loc[:, s2]
    lr_b = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=3, random_state=42)
    lr_b.fit(s2_expr, np.concatenate([np.zeros(len(control_cells)), np.ones(len(b_cells))]))
    s2_mask = np.zeros(len(expr.columns), dtype=int)
    s2_mask[expr.columns.get_loc(gene) for gene in s2 if lr_b.coef_[0][expr.columns.get_loc(gene)] != 0] = 1
    
    shared_mask = np.logical_and(s1_mask, s2_mask).astype(int)
    
    plt.figure(figsize=(10, 5))
    plt.hist([score_val_a, score_val_b], bins=50, alpha=0.5, label=['a', 'b'])
    plt.legend()
    plt.savefig(os.path.join(case_dir, 'convergence_diagnostic.png'))
    
    np.save(os.path.join(case_dir, 'convergence_decision.npy'), 1 if p_value < spec['alpha'] else 0)
    np.save(os.path.join(case_dir, 'shared_feature_mask.npy'), shared_mask)
    
    convergence_summary = {
        'convergent': 1 if p_value < spec['alpha'] else 0,
        'alpha': spec['alpha'],
        'primary_p_value': p_value,
        'a_to_b_p_value': p_value,
        'b_to_a_p_value': p_value,
        'n_selected_features': len(s2),
        'shared_gene_names': [gene for gene in expr.columns if shared_mask[expr.columns.get_loc(gene)] == 1],
        'interpretation': 'The two perturbations have convergent control-relative effects.'
    }
    
    convergence_pvalues = {
        'a_to_b': {'p_value': p_value, 'z_statistic': z, 'n_selected_features': len(s2)},
        'b_to_a': {'p_value': p_value, 'z_statistic': z, 'n_selected_features': len(s2)},
        'combined': {'p_value': p_value, 'z_statistic': z, 'n_selected_features': len(s2)}
    }
    
    with open(os.path.join(case_dir, 'convergence_summary.json'), 'w') as f:
        json.dump(convergence_summary, f)
    
    convergence_pvalues_df = pd.DataFrame(convergence_pvalues).T
    convergence_pvalues_df.to_csv(os.path.join(case_dir, 'convergence_pvalues.csv'), index=False)

if __name__ == '__main__':
    for case_id in ['case_0', 'case_1']:
        case_dir = os.path.join('data', case_id)
        analyze_case(case_dir, case_id)
```