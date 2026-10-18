```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def load_case(case_dir):
    expr = pd.read_csv(f'{case_dir}/expression_matrix.csv', index_col=0)
    meta = pd.read_csv(f'{case_dir}/cell_metadata.csv')
    spec = pd.read_json(f'{case_dir}/analysis_spec.json')
    return expr, meta, spec

def cross_fit_test(expr, meta, spec, direction='a_to_b'):
    # Split into K folds
    K = 5
    folds = np.array_split(np.arange(len(meta)), K)
    
    # Initialize results
    results = []
    projection_directions = []
    
    for fold_idx in range(K):
        # Split into train and validation
        train_indices = [i for i in range(K) if i != fold_idx]
        train_cells = np.concatenate([folds[i] for i in train_indices])
        val_cells = folds[fold_idx]
        
        # Select candidate features
        if direction == 'a_to_b':
            group_labels = meta.loc[train_cells, 'condition'].map({'control': 0, 'perturbation_a': 1})
            X_train = expr.iloc[train_cells, :].values
            X_val = expr.iloc[val_cells, :].values
            Y_train = group_labels.values
            # Fit sparse logistic regression
            model = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=5)
            model.fit(X_train, Y_train)
            s1 = np.where(model.coef_[0] != 0)[0]
            
            # Fit second-stage model
            X_train_s1 = X_train[:, s1]
            Y_train_s1 = Y_train
            model2 = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=5)
            model2.fit(X_train_s1, Y_train_s1)
            s2 = np.where(model2.coef_[0] != 0)[0]
            s2_full = np.zeros(expr.shape[1], dtype=int)
            s2_full[s1] = s2
            
            # Compute projection direction
            X_control = expr.iloc[train_cells[meta.loc[train_cells, 'condition'] == 'control'], :].values
            X_control_s1 = X_control[:, s1]
            pca = PCA(n_components=1)
            v_hat = pca.fit_transform(X_control_s1)
            v_hat = np.zeros(expr.shape[1])
            v_hat[s1] = v_hat[0]
            n_eff = min(len(train_cells[meta.loc[train_cells, 'condition'] == 'control']), len(val_cells))
            w_n = n_eff ** (1/3)
            beta_proj = v_hat + w_n * model2.coef_[0]
            beta_proj /= np.linalg.norm(beta_proj)
            projection_directions.append(beta_proj)
            
            # Compute scores
            X_val_s1 = X_val[:, s1]
            scores = X_val_s1 @ beta_proj
            group = meta.loc[val_cells, 'condition'].map({'control': 0, 'perturbation_b': 1})
            T_j = np.mean(scores[group == 0]) - np.mean(scores[group == 1])
            var_j = np.var(scores[group == 0]) / len(scores[group == 0]) + np.var(scores[group == 1]) / len(scores[group == 1])
            results.append((T_j, var_j))
        
        elif direction == 'b_to_a':
            group_labels = meta.loc[train_cells, 'condition'].map({'control': 0, 'perturbation_b': 1})
            X_train = expr.iloc[train_cells, :].values
            X_val = expr.iloc[val_cells, :].values
            Y_train = group_labels.values
            # Fit sparse logistic regression
            model = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=5)
            model.fit(X_train, Y_train)
            s1 = np.where(model.coef_[0] != 0)[0]
            
            # Fit second-stage model
            X_train_s1 = X_train[:, s1]
            Y_train_s1 = Y_train
            model2 = LogisticRegressionCV(penalty='l1', solver='liblinear', Cs=10, cv=5)
            model2.fit(X_train_s1, Y_train_s1)
            s2 = np.where(model2.coef_[0] != 0)[0]
            s2_full = np.zeros(expr.shape[1], dtype=int)
            s2_full[s1] = s2
            
            # Compute projection direction
            X_control = expr.iloc[train_cells[meta.loc[train_cells, 'condition'] == 'control'], :].values
            X_control_s1 = X_control[:, s1]
            pca = PCA(n_components=1)
            v_hat = pca.fit_transform(X_control_s1)
            v_hat = np.zeros(expr.shape[1])
            v_hat[s1] = v_hat[0]
            n_eff = min(len(train_cells[meta.loc[train_cells, 'condition'] == 'control']), len(val_cells))
            w_n = n_eff ** (1/3)
            beta_proj = v_hat + w_n * model2.coef_[0]
            beta_proj /= np.linalg.norm(beta_proj)
            projection_directions.append(beta_proj)
            
            # Compute scores
            X_val_s1 = X_val[:, s1]
            scores = X_val_s1 @ beta_proj
            group = meta.loc[val_cells, 'condition'].map({'control': 0, 'perturbation_a': 1})
            T_j = np.mean(scores[group == 0]) - np.mean(scores[group == 1])
            var_j = np.var(scores[group == 0]) / len(scores[group == 0]) + np.var(scores[group == 1]) / len(scores[group == 1])
            results.append((T_j, var_j))
    
    # Align signs
    baseline = projection_directions[0]
    aligned_results = []
    for i, (T_j, var_j) in enumerate(results):
        dot = np.dot(projection_directions[i], baseline)
        if dot < 0:
            T_j *= -1
        aligned_results.append((T_j, var_j))
    
    # Aggregate
    T_bar = np.mean([t for t, _ in aligned_results])
    SE = np.sqrt(np.sum([v for _, v in aligned_results]) / len(aligned_results)**2)
    z = T_bar / SE
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return p_value, T_bar, SE

def main(case_dir):
    expr, meta, spec = load_case(case_dir)
    a_p, a_T, a_SE = cross_fit_test(expr, meta, spec, 'a_to_b')
    b_p, b_T, b_SE = cross_fit_test(expr, meta, spec, 'b_to_b')
    
    # Combine results
    primary_p = min(a_p, b_p)
    shared_mask = np.zeros(expr.shape[1], dtype=int)
    # Placeholder for actual shared gene selection logic
    shared_mask[:5] = 1
    
    # Output results
    result = {
        'convergent': 1 if primary_p < spec['alpha'] else 0,
        'primary_p_value': primary_p,
        'a_to_b_p_value': a_p,
        'b_to_a_p_value': b_p,
        'shared_gene_names': [f'G{idx:04d}' for idx in np.where(shared_mask)[0]],
        'interpretation': 'Convergent if p-value < alpha'
    }
    np.save(f'results/{case_dir.split("/")[-1]}/convergence_decision.npy', result['convergent'])
    np.save(f'results/{case_dir.split("/")[-1]}/shared_feature_mask.npy', shared_mask)
    pd.DataFrame({
        'direction': ['a_to_b', 'b_to_a', 'combined'],
        'p_value': [a_p, b_p, primary_p],
        'z_statistic': [a_T, b_T, T_bar],
        'n_selected_features': [len(s1), len(s2), len(s1)+len(s2)]
    }).to_csv(f'results/{case_dir.split("/")[-1]}/convergence_pvalues.csv', index=False)
    plt.hist([a_T, b_T], bins=20)
    plt.savefig(f'results/{case_dir.split("/")[-1]}/convergence_diagnostic.png')
    with open(f'results/{case_dir.split("/")[-1]}/convergence_summary.json', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main('data/case_0')
    main('data/case_1')
```