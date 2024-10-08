import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from data_generating_process import create_data_generate_process
from ate_estimate import calculate_ate
from utilities import plot_propensity_score_distribution,estimate_propensity_score


# Set the seed for reproducibility
np.random.seed(42)

n_iterations = 10

# Iteration loop for ATE estimation
ate_with_xi_list = []
ate_without_xi_list = []
ground_truth_ate_list = []

for _ in range(n_iterations):
    
    X, T, Y, Y1, Y0, prob = create_data_generate_process(mode ='mode_1')
    
    scaler = StandardScaler()
    X_scaled_with = scaler.fit_transform(X)
    X_scaled_without = scaler.fit_transform(X[:, 1:])

    # estimate propensity scores
    propensity_scores_with = estimate_propensity_score(X_scaled_with, T)
    propensity_scores_without = estimate_propensity_score(X_scaled_without, T)
        
    # ATE estimation
    method = 'T-Learner'  # or 'IPW'
    ate_with_xi, ate_without_xi = calculate_ate(
        method, T, Y, X_scaled_with, X_scaled_without, propensity_scores_with, propensity_scores_without
    )
    
    # Store results for this iteration
    ate_with_xi_list.append(ate_with_xi)
    ate_without_xi_list.append(ate_without_xi)
    ground_truth_ate_list.append(np.mean(Y1 - Y0))

# Calculate average ATE and standard deviation over all iterations
avg_ate_with_xi = np.mean(ate_with_xi_list)
std_ate_with_xi = np.std(ate_with_xi_list)

avg_ate_without_xi = np.mean(ate_without_xi_list)
std_ate_without_xi = np.std(ate_without_xi_list)

absolute_ate_error_with_xi = np.mean([abs(ate - gt) for ate, gt in zip(ate_with_xi_list, ground_truth_ate_list)])
absolute_ate_error_without_xi = np.mean([abs(ate - gt) for ate, gt in zip(ate_without_xi_list, ground_truth_ate_list)])

avg_ground_truth_ate = np.mean(ground_truth_ate_list)
std_ground_truth_ate = np.std(ground_truth_ate_list)


# Plot Propensity Score Distributions (optional)
plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without,T)

# Display results
print(f"Averaged results over {n_iterations} iterations:")
print(f"Model: With x_i, ATE: {avg_ate_with_xi:.4f}, ATE Std: {std_ate_with_xi:.4f}")
print(f"Model: Without x_i, ATE: {avg_ate_without_xi:.4f}, ATE Std: {std_ate_without_xi:.4f}")
print(f"Ground Truth ATE: {avg_ground_truth_ate:.4f}, ATE Std: {std_ground_truth_ate:.4f}")

print(f"Average Absolute ATE Error (With x_i): {absolute_ate_error_with_xi:.4f}")
print(f"Average Absolute ATE Error (Without x_i): {absolute_ate_error_without_xi:.4f}")



