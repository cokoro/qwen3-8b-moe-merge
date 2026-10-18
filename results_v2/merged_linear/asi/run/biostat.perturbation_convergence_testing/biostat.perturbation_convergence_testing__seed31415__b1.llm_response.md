```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import json

def load_data(case_dir):
    expr_path = os.path.join(case_dir, 'expression_matrix.csv')
    meta_path = os.path.join(case_dir, 'cell_metadata.csv')
    spec_path = os.path.join(case_dir, 'analysis_spec.json')
    
    expr = pd.read_csv(expr_path, index_col=0)
    meta = pd.read_csv(meta_path, index_col=0)
    spec = json.load(open(spec_path))
    
    return expr, meta, spec

def cross_fitted_test(expr, meta, spec, direction='a_to_b'):
    conditions = spec['condition_labels']
    control = conditions['control']
    perturbation_a = conditions['perturbation_a']
    perturbation_b = conditions['perturbation_b']
    
    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    all_results = []
    
    for train_idx, valid_idx in kf.split(expr.index):
        train_cells = expr.index[train_idx]
        valid_cells = expr.index[valid_idx]
        
        # Prepare training and validation sets
        train_meta = meta.loc[train_cells]
        valid_meta = meta.loc[valid_cells]
        
        train_control_cells = train_cells[train_meta['condition'] == control]
        train_perturbation_a_cells = train_cells[train_meta['condition'] == perturbation_a]
        train_perturbation_b_cells = train_cells[train_meta['condition'] == perturbation_b]
        
        valid_control_cells = valid_cells[valid_meta['condition'] == control]
        valid_perturbation_b_cells = valid_cells[valid_meta['condition'] == perturbation_b]
        
        # Direction A_to_B
        if direction == 'a_to_b':
            train_y = train_meta['condition'].map({control: 0, perturbation_a: 1}).values
            train_X = expr.loc[train_cells].values
            train_y = train_y[train_cells]
            
            # Step 1: Select candidate features for perturbation_a vs control
            lr = LogisticRegression(penalty='l1', solver='liblinear', C=1e5, max_iter=1000)
            lr.fit(train_X, train_y)
            s1 = np.where(lr.coef_[0] != 0)[0]
            
            # Step 2: Select candidate features for perturbation_b vs control within s1
            train_X_s1 = train_X[:, s1]
            train_y_b = train_meta['condition'].map({control: 0, perturbation_b: 1}).values
            train_y_b = train_y_b[train_cells]
            
            lr_b = LogisticRegression(penalty='l1', solver='liblinear', C=1e5, max_iter=1000)
            lr_b.fit(train_X_s1, train_y_b)
            s2 = np.where(lr_b.coef_[0] != 0)[0]
            shared_features = s1[s2]
            
            # Step 3: Compute projection direction
            train_X_s1 = train_X[:, shared_features]
            scaler = StandardScaler()
            train_X_s1_scaled = scaler.fit_transform(train_X_s1)
            
            pca = PCA(n_components=1)
            v_hat = pca.fit_transform(train_X_s1_scaled)
            v_hat = v_hat.flatten()
            
            n_effective = min(len(train_perturbation_b_cells), len(train_control_cells))
            w_n = n_effective ** (1/3)
            beta2_hat = lr_b.coef_[0]
            beta_proj = v_hat + w_n * beta2_hat
            v = beta_proj / np.linalg.norm(beta_proj)
            
            # Step 4: Compute scores on validation cells
            valid_X_s1 = expr.loc[valid_cells, shared_features].values
            valid_X_s1_scaled = scaler.transform(valid_X_s1)
            scores = valid_X_s1_scaled @ v
            
            # Step 5: Compute fold statistic
            scores_control = scores[valid_meta['condition'] == control]
            scores_perturbation_b = scores[valid_meta['condition'] == perturbation_b]
            T = np.mean(scores_control) - np.mean(scores_perturbation_b)
            
            # Step 6: Compute fold variance
            Var = np.var(scores_control) / len(scores_control) + np.var(scores_perturbation_b) / len(scores_perturbation_b)
            
            # Step 7: Align signs
            if len(all_results) == 0:
                v_baseline = v
            else:
                as_array = np.array(all_results[-1]['projection'])
                dot_product = as_array @ v_baseline
                if dot_product < 0:
                    T *= -1
                v = v_baseline
            
            all_results.append({
                'projection': v,
                'T': T,
                'Var': Var,
                'shared_features': shared_features.tolist(),
                's1': s1.tolist(),
                's2': s2.tolist()
            })
    
    # Aggregate results
    T_bar = np.mean([res['T'] for res in all_results])
    SE = np.sqrt(sum([res['Var'] for res in all_results]) / len(all_results)**2)
    z = T_bar / SE
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return p_value, all_results, shared_features

def main(case_dir):
    expr, meta, spec = load_data(case_dir)
    case_name = os.path.basename(case_dir)
    
    p_a_to_b, results_a_to_b, shared_features_a_to_b = cross_fitted_test(expr, meta, spec, direction='a_to_b')
    p_b_to_a, results_b_to_a, shared_features_b_to_a = cross_fitted_test(expr, meta, spec, direction='b_to_a')
    
    shared_gene_names = [f'G{idx:04d}' for idx in np.union1d(shared_features_a_to_b, shared_features_b_to_a)]
    
    primary_p_value = min(p_a_to_b, p_b_to_a)
    
    # Write output files
    os.makedirs(os.path.join('results', case_name), exist_ok=True)
    summary = {
        'convergent': 1 if primary_p_value < spec['alpha'] else 0,
        'alpha': spec['alpha'],
        'primary_p_value': primary_p_value,
        'a_to_b_p_value': p_a_to_b,
        'b_to_a_p_value': p_b_to_a,
        'n_selected_features': len(shared_gene_names),
        'shared_gene_names': shared_gene_names,
        'interpretation': 'Convergent' if primary_p_value < spec['alpha'] else 'Not convergent'
    }
    with open(os.path.join('results', case_name, 'convergence_summary.json'), 'w') as f:
        json.dump(summary, f)
    
    pvalues = {
        'a_to_b': p_a_to_b,
        'b_to_a': p_b_to_a,
        'combined': primary_p_value
    }
    pvalues_df = pd.DataFrame(pvalues.items(), columns=['direction', 'p_value'])
    pvalues_df['z_statistic'] = [0, 0, 0]  # Placeholder
    pvalues_df['n_selected_features'] = [len(shared_features_a_to_b), len(shared_features_b_to_a), len(shared_gene_names)]
    pvalues_df.to_csv(os.path.join('results', case_name, 'convergence_pvalues.csv'), index=False)
    
    mask = np.zeros(1941, dtype=np.int64)
    for idx in shared_gene_names:
        mask[int(idx[1:])] = 1
    np.save(os.path.join('results', case_name, 'shared_feature_mask.npy'), mask)
    
    # Diagnostic plot
    plt.figure(figsize=(10, 6))
    sns.histplot([res['T'] for res in results_a_to_b], kde=True, label='a_to_b')
    sns.histplot([res['T'] for res in results_b_to_a], kde=True, label='b_to_a')
    plt.title('Cross-Fitted Fold Statistics')
    plt.xlabel('Fold Statistic')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join('results', case_name, 'convergence_diagnostic.png'))
    plt.close()

if __name__ == '__main__':
    for case in ['case_0', 'case_1']:
        case_dir = os.path.join('data', case)
        main(case_dir)
```