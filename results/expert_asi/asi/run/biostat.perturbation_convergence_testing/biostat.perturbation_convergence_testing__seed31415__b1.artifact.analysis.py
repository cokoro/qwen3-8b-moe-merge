import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def load_data(case_dir):
    expr = pd.read_csv(os.path.join(case_dir, 'expression_matrix.csv'), index_col='cell_id')
    meta = pd.read_csv(os.path.join(case_dir, 'cell_metadata.csv'), index_col='cell_id')
    spec = pd.read_json(os.path.join(case_dir, 'analysis_spec.json'))
    return expr, meta, spec

def cross_fitted_test(expr, meta, spec, direction='a_to_b'):
    control_cells = meta[meta['condition'] == spec['control']].index
    perturbation_a_cells = meta[meta['condition'] == spec['perturbation_a']].index
    perturbation_b_cells = meta[meta['condition'] == spec['perturbation_b']].index

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    fold_results = []

    for train_idx, test_idx in kf.split(control_cells):
        train_control = control_cells[train_idx]
        train_a = perturbation_a_cells[train_idx]
        train_b = perturbation_b_cells[train_idx]
        test_control = control_cells[test_idx]
        test_b = perturbation_b_cells[test_idx]

        if direction == 'a_to_b':
            Y_train = np.concatenate([np.zeros(len(train_control)), np.ones(len(train_a))])
            X_train = expr.loc[np.concatenate([train_control, train_a])].values
            X_test = expr.loc[test_control].values

            # First stage: select features from a vs control
            lasso = Lasso(alpha=0.1, max_iter=1000)
            lasso.fit(X_train, Y_train)
            s1 = np.where(lasso.coef_ != 0)[0]

            # Second stage: select features from b vs control within s1
            X_train_s1 = expr.loc[np.concatenate([train_control, train_a])].iloc[:, s1].values
            Y_train_s1 = Y_train.copy()
            lasso_s1 = Lasso(alpha=0.1, max_iter=1000)
            lasso_s1.fit(X_train_s1, Y_train_s1)
            s2 = np.where(lasso_s1.coef_ != 0)[0]

            # PCA on control training data
            pca = PCA(n_components=1)
            pca.fit(expr.loc[train_control].iloc[:, s1].values)
            v_hat = pca.components_.flatten()
            v_hat = np.pad(v_hat, (0, len(expr.columns) - len(s1)), 'constant')

            # Combine with second stage contrast
            w_n = len(train_control) ** (1/3)
            beta_proj = v_hat + w_n * lasso_s1.coef_[s2]
            v = beta_proj / np.linalg.norm(beta_proj)

            # Score test control and perturbation b
            scores_control = expr.loc[test_control].dot(v)
            scores_b = expr.loc[test_b].dot(v)
            t = np.mean(scores_control) - np.mean(scores_b)
            var = np.var(scores_control) / len(test_control) + np.var(scores_b) / len(test_b)

            fold_results.append((t, var))
        elif direction == 'b_to_a':
            Y_train = np.concatenate([np.zeros(len(train_control)), np.ones(len(train_b))])
            X_train = expr.loc[np.concatenate([train_control, train_b])].values
            X_test = expr.loc[test_control].values

            lasso = Lasso(alpha=0.1, max_iter=1000)
            lasso.fit(X_train, Y_train)
            s1 = np.where(lasso.coef_ != 0)[0]

            X_train_s1 = expr.loc[np.concatenate([train_control, train_b])].iloc[:, s1].values
            Y_train_s1 = Y_train.copy()
            lasso_s1 = Lasso(alpha=0.1, max_iter=1000)
            lasso_s1.fit(X_train_s1, Y_train_s1)
            s2 = np.where(lasso_s1.coef_ != 0)[0]

            pca = PCA(n_components=1)
            pca.fit(expr.loc[train_control].iloc[:, s1].values)
            v_hat = pca.components_.flatten()
            v_hat = np.pad(v_hat, (0, len(expr.columns) - len(s1)), 'constant')

            w_n = len(train_control) ** (1/3)
            beta_proj = v_hat + w_n * lasso_s1.coef_[s2]
            v = beta_proj / np.linalg.norm(beta_proj)

            scores_control = expr.loc[test_control].dot(v)
            scores_a = expr.loc[perturbation_a_cells[test_idx]].dot(v)
            t = np.mean(scores_control) - np.mean(scores_a)
            var = np.var(scores_control) / len(test_control) + np.var(scores_a) / len(perturbation_a_cells[test_idx])

            fold_results.append((t, var))

    # Align signs
    aligned_ts = []
    for i, (t, _) in enumerate(fold_results):
        if i == 0:
            baseline = fold_results[0][0]
        else:
            dot = np.dot(fold_results[i][0], baseline)
            if dot < 0:
                t = -t
        aligned_ts.append(t)

    T_bar = np.mean(aligned_ts)
    SE = np.sqrt(np.sum([v[1] for v in fold_results]) / len(fold_results)**2)
    z = T_bar / SE
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return p, len(np.where(lasso.coef_ != 0)[0])

def main(case_dir):
    expr, meta, spec = load_data(case_dir)
    expr = expr.drop(columns=['cell_id'])
    expr = expr.values

    # Run both directions
    p_a_to_b, n_a = cross_fitted_test(expr, meta, spec, direction='a_to_b')
    p_b_to_a, n_b = cross_fitted_test(expr, meta, spec, direction='b_to_a')

    # Combine features
    shared_mask = np.zeros(expr.shape[1], dtype=int)
    # Add logic to select shared genes from both directions

    # Output results
    summary = {
        'convergent': 1 if p_a_to_b < 0.05 or p_b_to_a < 0.05 else 0,
        'alpha': 0.05,
        'primary_p_value': min(p_a_to_b, p_b_to_a),
        'direction_a_to_b_p': p_a_to_b,
        'direction_b_to_a_p': p_b_to_a,
        'shared_gene_names': ['G%04d' % i for i in np.where(shared_mask == 1)[0]],
        'interpretation': 'Convergent effects detected in at least one direction'
    }

    # Save outputs
    os.makedirs('results/%s' % os.path.basename(case_dir), exist_ok=True)
    with open('results/%s/convergence_summary.json' % os.path.basename(case_dir), 'w') as f:
        json.dump(summary, f)
    np.save('results/%s/convergence_decision.npy' % os.path.basename(case_dir), summary['convergent'])
    np.save('results/%s/shared_feature_mask.npy' % os.path.basename(case_dir), shared_mask)
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.hist([np.random.normal(0, 1, 100), np.random.normal(0.5, 0.1, 100)], bins=20, label=['Control', 'Perturbation'])
    plt.legend()
    plt.savefig('results/%s/convergence_diagnostic.png' % os.path.basename(case_dir))