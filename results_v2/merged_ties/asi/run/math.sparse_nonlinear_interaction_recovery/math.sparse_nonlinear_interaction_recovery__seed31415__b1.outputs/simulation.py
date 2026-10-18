import numpy as np
import pandas as pd
import json
import os
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

def load_data():
    design_matrix = np.load('./data/design_matrix.npy', allow_pickle=True)
    observations = np.load('./data/observations.npy', allow_pickle=True)
    nuisance_covariates = np.load('./data/nuisance_covariates.npy', allow_pickle=True)
    environment_id = np.load('./data/environment_id.npy', allow_pickle=True)
    group_structure = json.load(open('./data/group_structure.json', 'r'))
    return design_matrix, observations, nuisance_covariates, environment_id, group_structure

def preprocess_data(design_matrix, nuisance_covariates, environment_id):
    # Remove nuisance/environment drift
    group_assignments = np.array([int(x.split('_')[0]) for x in environment_id])
    group_means = np.array([np.mean(design_matrix[group_assignments == g], axis=0) for g in np.unique(group_assignments)])
    centered_design = design_matrix - group_means[group_assignments]
    
    # Remove nuisance covariates
    nuisance_mask = np.isin(np.arange(design_matrix.shape[1]), np.array(nuisance_covariates, dtype=int))
    centered_design = centered_design[:, ~nuisance_mask]
    return centered_design

def estimate_nonlinear_effects(design_matrix, observations):
    # Use polynomial features and Lasso to estimate nonlinear main effects
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(design_matrix)
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_poly, observations)
    beta_hat = lasso.coef_
    return beta_hat

def identify_nonlinear_features(beta_hat, threshold=0.01):
    # Use absolute value thresholding to identify features with nonlinear main effects
    support_indicator = np.abs(beta_hat) > threshold
    return support_indicator

def estimate_interaction_effects(design_matrix, observations, beta_hat):
    # Use pairwise products and Lasso to estimate interaction effects
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(design_matrix)
    interaction_mask = np.abs(beta_hat) < 0.01
    X_interaction = X_poly[:, interaction_mask]
    lasso = Lasso(alpha=0.1, max_iter=10000)
    lasso.fit(X_interaction, observations)
    interaction_weight = lasso.coef_
    return interaction_weight

def main():
    design_matrix, observations, nuisance_covariates, environment_id, group_structure = load_data()
    centered_design = preprocess_data(design_matrix, nuisance_covariates, environment_id)
    beta_hat = estimate_nonlinear_effects(centered_design, observations)
    support_indicator = identify_nonlinear_features(beta_hat)
    interaction_weight = estimate_interaction_effects(centered_design, observations, beta_hat)

    # Save outputs
    np.save('./beta_hat.npy', beta_hat)
    np.save('./support_indicator.npy', support_indicator)
    np.save('./interaction_weight.npy', interaction_weight)

if __name__ == '__main__':
    main()