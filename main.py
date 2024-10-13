import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from data_generating_process import create_data_generate_process
from ate_estimate import calculate_ate
from utilities import plot_propensity_score_distribution,estimate_propensity_score,calculate_ate_statistics


# Set the seed for reproducibility
np.random.seed(42)

n_iterations = 200

# Iteration loop for ATE estimation
ate_with_xi_list = []
ate_without_xi_list = []
ground_truth_ate_list = []
naive_ate_list = []

for _ in range(n_iterations):
    
    X, T, Y, Y1, Y0, prob = create_data_generate_process(mode ='mode_poly_all')
    #X, T, Y, Y1, Y0, prob = create_data_generate_process2(mode ='mode_4')
    
    scaler = StandardScaler()
    X_scaled_with = X #scaler.fit_transform(X)
    X_scaled_without = X[:, 1:]  #scaler.fit_transform(X[:, 1:])

    # estimate propensity scores
    propensity_scores_with = estimate_propensity_score(X_scaled_with, T)
    propensity_scores_without = estimate_propensity_score(X_scaled_without, T)
        
    # ATE estimation
    method = 'T-Learner'  # or 'IPW'
    estimator_name = 'PolynomialRegression2'  # Change this to 'LinearRegression','PolynomialRegression2', 'PolynomialRegression3', 'CatBoost', 'XGBoost', or 'GradientBoosting'
    ate_with_xi, ate_without_xi = calculate_ate(
        method, T, Y, X_scaled_with, X_scaled_without, propensity_scores_with, propensity_scores_without, estimator_name
    )
    
    # Store results for this iteration
    ate_with_xi_list.append(ate_with_xi)
    ate_without_xi_list.append(ate_without_xi)
    ground_truth_ate_list.append(np.mean(Y1 - Y0))
    naive_ate_list.append(np.mean(Y[T == 1]) - np.mean(Y[T == 0]))    

# Plot Propensity Score Distributions (optional)
plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without,T)

# Calculate average ATE and ATE error
statistics_with_xi = calculate_ate_statistics(ate_with_xi_list, ground_truth_ate_list, n_iterations)
statistics_without_xi = calculate_ate_statistics(ate_without_xi_list, ground_truth_ate_list, n_iterations)
statistics_naive = calculate_ate_statistics(naive_ate_list, ground_truth_ate_list, n_iterations)

# Calculate ground truth statistics
avg_ground_truth_ate = np.mean(ground_truth_ate_list)
std_ground_truth_ate = np.std(ground_truth_ate_list)
sem_ground_truth_ate = std_ground_truth_ate / np.sqrt(n_iterations)

# Display results
print(f"Averaged results over {n_iterations} iterations:")
print(f"Ground Truth ATE:                {avg_ground_truth_ate:.4f} ± {sem_ground_truth_ate:.4f}")
print(f"=====================================================================")
print(f"Naive mean ATE:                  {statistics_naive['avg_ate']:.4f} ± {statistics_naive['sem_ate']:.4f}")
print(f"Model: With x_i, mean ATE:       {statistics_with_xi['avg_ate']:.4f} ± {statistics_with_xi['sem_ate']:.4f}")
print(f"Model: Without x_i, mean ATE:    {statistics_without_xi['avg_ate']:.4f} ± {statistics_without_xi['sem_ate']:.4f}")
print(f"=====================================================================")
print(f"Average ATE Error (Naive):       {statistics_naive['ate_error']:.4f} ± {statistics_naive['sem_ate_error']:.4f}")
print(f"Average ATE Error (With x_i):    {statistics_with_xi['ate_error']:.4f} ± {statistics_with_xi['sem_ate_error']:.4f}")
print(f"Average ATE Error (Without x_i): {statistics_without_xi['ate_error']:.4f} ± {statistics_without_xi['sem_ate_error']:.4f}")













