```python
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.stats import norm

def load_data(case_dir):
    expr_path = os.path.join(case_dir, "expression_matrix.csv")
    meta_path = os.path.join(case_dir, "cell_metadata.csv")
    spec_path = os.path.join(case_dir, "analysis_spec.json")
    
    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path)
    spec = json.load(open(spec_path))
    
    conditions = spec['conditions']
    control = meta[meta['condition'] == conditions['control']]
    perturbation_a = meta[meta['condition'] == conditions['perturbation_a']]
    perturbation_b = meta[meta['condition'] == conditions['perturbation_b']]
    
    expr_control = expr.loc[control['cell_id']]
    expr_a = expr.loc[perturbation_a['cell_id']]
    expr_b = expr.loc[perturbation_b['cell_id']]
    
    return expr_control, expr_a, expr_b, control, perturbation_a, perturbation_b, spec

def perform_analysis(expr_control, expr_a, expr_b, control, perturbation_a, perturbation_b, spec):
    K = 5
    results = {'a_to_b': {}, 'b_to_a': {}}
    shared_gene_mask = np.zeros(expr_control.shape[1], dtype=int)
    fold_results = []

    for direction in ['a_to_b', 'b_to_a']:
        fold_stats = []
        fold_vars = []
        fold_directions = []

        for fold_idx in range(K):
            kf = KFold(n_splits=K)
            train_indices = []
            val_indices = []

            for train_idx, val_idx in kf.split(control):
                train_indices.append((train_idx, val_idx))
                break

            train_control, val_control = control.iloc[train_indices[fold_idx][0]], control.iloc[train_indices[fold_idx][1]]
            train_a, val_a = perturbation_a.iloc[train_indices[fold_idx][0]], perturbation_a.iloc[train_indices[fold_idx][1]]
            train_b, val_b = perturbation_b.iloc[train_indices[fold_idx][0]], perturbation_b.iloc[train_indices[fold_idx][1]]

            train_expr_control = expr_control.loc[train_control['cell_id']]
            train_expr_a = expr_a.loc[train_a['cell_id']]
            train_expr_b = expr_b.loc[train_b['cell_id']]
            val_expr_control = expr_control.loc[val_control['cell_id']]
            val_expr_a = expr_a.loc[val_a['cell_id']]
            val_expr_b = expr_b.loc[val_b['cell_id']]

            if direction == 'a_to_b':
                y_train = np.zeros(len(train_expr_control) + len(train_expr_a))
                y_train[len(train_expr_control):] = 1
                X_train = pd.concat([train_expr_control, train_expr_a], axis=0)
                y_train = y_train.astype(int)
                X_train = X_train.values
            else:
                y_train = np.zeros(len(train_expr_control) + len(train_expr_b))
                y_train[len(train_expr_control):] = 1
                X_train = pd.concat([train_expr_control, train_expr_b], axis=0)
                y_train = y_train.astype(int)
                X_train = X_train.values

            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
            model.fit(X_train, y_train)
            s1 = np.where(model.coef_[0] != 0)[0]

            if direction == 'a_to_b':
                X_train_restricted = X_train[:, s1]
                y_train_restricted = y_train
                model2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
                model2.fit(X_train_restricted, y_train_restricted)
                s2 = np.where(model2.coef_[0] != 0)[0]
                s2_full = np.zeros(X_train.shape[1])
                s2_full[s1[s2]] = 1
            else:
                X_train_restricted = X_train[:, s1]
                y_train_restricted = y_train
                model2 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
                model2.fit(X_train_restricted, y_train_restricted)
                s2 = np.where(model2.coef_[0] != 0)[0]
                s2_full = np.zeros(X_train.shape[1])
                s2_full[s1[s2]] = 1

            X_train_control = expr_control.loc[train_control['cell_id']].values
            X_train_control_restricted = X_train_control[:, s1]
            pca = PCA(n_components=1)
            v_hat = pca.fit_transform(X_train_control_restricted)
            v_hat = np.squeeze(v_hat)
            v_hat = np.zeros(X_train.shape[1])
            v_hat[s1] = v_hat

            n_effective = min(len(train_expr_control), len(train_expr_b))
            w_n = n_effective ** (1/3)
            beta_proj = v_hat + w_n * model2.coef_[0]
            beta_proj = beta_proj / np.linalg.norm(beta_proj)

            score_control = val_expr_control.values @ beta_proj
            score_b = val_expr_b.values @ beta_proj

            T_j = np.mean(score_control) - np.mean(score_b)
            Var_j = np.var(score_control) / len(score_control) + np.var(score_b) / len(score_b)
            fold_stats.append(T_j)
            fold_vars.append(Var_j)

            fold_directions.append(beta_proj)

        T_bar = np.mean(fold_stats)
        SE = np.sqrt(np.sum(fold_vars) / K**2)
        z = T_bar / SE
        p_value = 2 * norm.cdf(-abs(z))

        results[direction]['p_value'] = p_value
        results[direction]['z_statistic'] = z
        results[direction]['n_selected_features'] = len(s1)

        if direction == 'a_to_b':
            shared_gene_mask[s1[s2]] = 1
        else:
            shared_gene_mask[s1[s2]] = 1

    primary_p_value = min(results['a_to_b']['p_value'], results['b_to_a']['p_value'])
    convergent = 1 if primary_p_value < spec['alpha'] else 0
    directional_p_values = {
        'a_to_b': results['a_to_b']['p_value'],
        'b_to_b': results['b_to_a']['p_value'],
        'combined': primary_p_value
    }

    return results, shared_gene_mask, convergent, directional_p_values

def save_results(case_dir, results, shared_gene_mask, convergent, directional_p_values, spec):
    os.makedirs(case_dir, exist_ok=True)

    convergence_summary = {
        'convergent': convergent,
        'alpha': spec['alpha'],
        'primary_p_value': directional_p_values['combined'],
        'a_to_b_p_value': directional_p_values['a_to_b'],
        'b_to_a_p_value': directional_p_values['b_to_a'],
        'n_selected_features_a_to_b': results['a_to_b']['n_selected_features'],
        'n_selected_features_b_to_a': results['b_to_a']['n_selected_features'],
        'shared_gene_names': [f'G{str(i).zfill(4)}' for i in np.where(shared_gene_mask == 1)[0]],
        'interpretation': 'Convergent' if convergent else 'Not Convergent'
    }

    with open(os.path.join(case_dir, 'convergence_summary.json'), 'w') as f:
        json.dump(convergence_summary, f)

    convergence_pvalues = pd.DataFrame({
        'direction': ['a_to_b', 'b_to_a', 'combined'],
        'p_value': [directional_p_values['a_to_b'], directional_p_values['b_to_a'], directional_p_values['combined']],
        'z_statistic': [results['a_to_b']['z_statistic'], results['b_to_a']['z_statistic'], 0]
    })
    convergence_pvalues.to_csv(os.path.join(case_dir, 'convergence_pvalues.csv'), index=False)

    np.save(os.path.join(case_dir, 'convergence_decision.npy'), convergent)
    np.save(os.path.join(case_dir, 'shared_feature_mask.npy'), shared_gene_mask)

    plt.figure(figsize=(10, 6))
    plt.hist([results['a_to_b']['n_selected_features'], results['b_to_a']['n_selected_features']], bins=range(0, 2000, 100))
    plt.xlabel('Number of Selected Features')
    plt.ylabel('Frequency')
    plt.title('Selected Features Distribution')
    plt.savefig(os.path.join(case_dir, 'convergence_diagnostic.png'))
    plt.close()

def main():
    for case in ['case_0', 'case_1']:
        case_dir = os.path.join('data', case)
        expr_control, expr_a, expr_b, control, perturbation_a, perturbation_b, spec = load_data(case_dir)
        results, shared_gene_mask, convergent, directional_p_values = perform_analysis(expr_control, expr_a, expr_b, control, perturbation_a, perturbation_b, spec)
        save_results(os.path.join('results', case), results, shared_gene_mask, convergent, directional_p_values, spec)

if __name__ == '__main__':
    main()
```