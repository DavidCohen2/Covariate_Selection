import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from data_generating_process import create_data_generate_process
from ate_estimate import calculate_ate
from utilities import plot_ate_error_vs_x_i_outcome_effect_weight


# Set the seed for reproducibility
np.random.seed(42)

n_iterations = 10
n_samples = 1000

# Define a range of x_i_outcome_effect_weight values
x_i_outcome_effect_weights = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 5]

# Lists to store results for the new figure
mean_ate_error_with_xi_list = []
mean_ate_error_without_xi_list = []
std_ate_error_with_xi_list = []
std_ate_error_without_xi_list = []

X, T, prob, weights = create_data_generate_process(mode='mode_folds_step1')
scaler = StandardScaler()
X_scaled_with = scaler.fit_transform(X)
X_scaled_without = scaler.fit_transform(X[:, 1:])

# Loop over different x_i_outcome_effect_weights
for x_i_outcome_effect_weight in x_i_outcome_effect_weights:
    ate_with_xi_list = []
    ate_without_xi_list = []
    ground_truth_ate_list = []

    Y, Y1, Y0 = create_data_generate_process(mode='mode_folds_step2', X=X, T=T, weights=weights, x_i_outcome_effect_weight=x_i_outcome_effect_weight)

    for _ in range(n_iterations):
        most_idx = np.random.choice(n_samples, int(n_samples * 0.8), replace=False)
        X_scaled_with_fold = X_scaled_with[most_idx]
        X_scaled_without_fold = X_scaled_without[most_idx]
        T_fold = T[most_idx]
        Y_fold = Y[most_idx]

        # Fit logistic regression models for propensity scores
        propensity_model_with = LogisticRegression().fit(X_scaled_with_fold, T_fold)
        propensity_model_without = LogisticRegression().fit(X_scaled_without_fold, T_fold)
        
        propensity_scores_with = propensity_model_with.predict_proba(X_scaled_with_fold)[:, 1]
        propensity_scores_without = propensity_model_without.predict_proba(X_scaled_without_fold)[:, 1]
        
        # ATE estimation
        method = 'T-Learner'  # or 'IPW'
        ate_with_xi, ate_without_xi = calculate_ate(
            method, T_fold, Y_fold, X_scaled_with_fold, X_scaled_without_fold, propensity_scores_with, propensity_scores_without
        )
        
        # Store results for this iteration
        ate_with_xi_list.append(ate_with_xi)
        ate_without_xi_list.append(ate_without_xi)
        ground_truth_ate_list.append(np.mean(Y1 - Y0))

    # Calculate absolute ATE errors
    ate_errors_with_xi = [abs(ate - gt) for ate, gt in zip(ate_with_xi_list, ground_truth_ate_list)]
    mean_ate_error_with_xi = np.mean(ate_errors_with_xi)
    std_ate_error_with_xi = np.std(ate_errors_with_xi)
    ate_errors_without_xi = [abs(ate - gt) for ate, gt in zip(ate_without_xi_list, ground_truth_ate_list)]
    mean_ate_error_without_xi = np.mean(ate_errors_without_xi)
    std_ate_error_without_xi = np.std(ate_errors_without_xi)
    
    mean_ate_error_with_xi_list.append(mean_ate_error_with_xi)
    std_ate_error_with_xi_list.append(std_ate_error_with_xi)
    mean_ate_error_without_xi_list.append(mean_ate_error_without_xi)
    std_ate_error_without_xi_list.append(std_ate_error_without_xi)

plot_ate_error_vs_x_i_outcome_effect_weight(x_i_outcome_effect_weights, mean_ate_error_with_xi_list, mean_ate_error_without_xi_list)

