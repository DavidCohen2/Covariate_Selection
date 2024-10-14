import numpy as np

from sklearn.preprocessing import StandardScaler
from data_generating_process import create_data_generate_process
from utilities import plot_propensity_score_distribution,estimate_propensity_score


# Set the seed for reproducibility
np.random.seed(42)

# Iteration loop for ATE estimation
ate_with_xi_list = []
ate_without_xi_list = []
ground_truth_ate_list = []
naive_ate_list = []

X, T, prob, weights = create_data_generate_process(mode='mode_folds_simple_step1', n_samples_outside=10000)
Y, Y1, Y0 = create_data_generate_process(mode='mode_folds_simple_step2', X=X, T=T, weights=weights, x_i_outcome_effect_weight=0.05)

scaler = StandardScaler()
X_scaled_with = X #scaler.fit_transform(X)
X_scaled_without = X[:, 1:]  #scaler.fit_transform(X[:, 1:])

# estimate propensity scores
propensity_scores_with = estimate_propensity_score(X_scaled_with, T)
propensity_scores_without = estimate_propensity_score(X_scaled_without, T)

# Plot Propensity Score Distributions (optional)
plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without,T)











