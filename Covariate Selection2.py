import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

# Set the seed for reproducibility
seed = 42
rng = np.random.default_rng(seed)

# Parameters
n_samples = 100
d_cov = 3  # Total number of covariates including x_i
treatment_effect = 3
n_iterations = 100  # Number of iterations for simulation

x_i_selection_weight = 10
x_i_outcome_effect_weight = 0.2
x_i_mean = 0

# Helper functions
def simulate_data(n_samples, d_cov, x_i_mean):
    X = np.random.normal(0, 1, (n_samples, d_cov))
    X[:, 0] += x_i_mean
    return X

def calculate_weights(d_cov):
    z = np.random.normal(0, 1, d_cov - 1)
    return z / np.sqrt(d_cov - 1)

def calculate_propensity_scores(X, weights, x_i_selection_weight):
    l = np.dot(X[:, 1:], weights) + x_i_selection_weight * X[:, 0]
    prob = 1 / (1 + np.exp(-l))
    T = np.random.binomial(1, prob)
    return T, prob

def simulate_outcomes(X, weights, T, treatment_effect, x_i_outcome_effect_weight):
    Y0 = np.dot(X[:, 1:], weights) + x_i_outcome_effect_weight * X[:, 0]**2
    Y1 = Y0 + treatment_effect 
    Y = T * Y1 + (1 - T) * Y0
    return Y, Y1, Y0

def estimate_ate_ipw(T, Y, propensity_score):
    weights = T / propensity_score + (1 - T) / (1 - propensity_score)
    weighted_outcome = weights * Y
    ate = np.mean(weighted_outcome[T == 1]) - np.mean(weighted_outcome[T == 0])
    return ate

# def estimate_ate_t_learner(T, Y, X):
#     model_treated = LinearRegression().fit(X[T == 1], Y[T == 1])
#     model_control = LinearRegression().fit(X[T == 0], Y[T == 0])
#     y_pred_treated = model_treated.predict(X)
#     y_pred_control = model_control.predict(X)
#     ate = np.mean(y_pred_treated - y_pred_control)
#     return ate
#RandomForestRegressor()
#GradientBoostingRegressor()
#LinearRegression()
def estimate_ate_t_learner(T, Y, X, estimator=GradientBoostingRegressor()):
    model_treated = estimator.__class__()  # Create a new instance of the estimator
    model_control = estimator.__class__()  # Create another new instance of the estimator
    
    model_treated.fit(X[T == 1], Y[T == 1])
    model_control.fit(X[T == 0], Y[T == 0])
    
    y_pred_treated = model_treated.predict(X)
    y_pred_control = model_control.predict(X)
    ate = np.mean(y_pred_treated - y_pred_control)
    return ate

def calculate_ate(method, T, Y, X_scaled_with, X_scaled_without, propensity_scores_with, propensity_scores_without):
    if method == 'IPW':
        ate_with_xi = estimate_ate_ipw(T, Y, propensity_scores_with)
        ate_without_xi = estimate_ate_ipw(T, Y, propensity_scores_without)
    elif method == 'T-Learner':
        ate_with_xi = estimate_ate_t_learner(T, Y, X_scaled_with)
        ate_without_xi = estimate_ate_t_learner(T, Y, X_scaled_without)
    else:
        raise ValueError("Method must be either 'IPW' or 'T-Learner'")
    
    return ate_with_xi, ate_without_xi

def plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without):
    plt.figure(figsize=(14, 6))
    for i, (scores, label) in enumerate(zip([propensity_scores_with, propensity_scores_without], ['With x_i', 'Without x_i'])):
        plt.subplot(1, 2, i+1)
        plt.hist(scores, bins=30, alpha=0.5, label=f'{label}')
        plt.title(f'Propensity Score Distribution ({label})')
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.show()

# Iteration loop for ATE estimation
ate_with_xi_list = []
ate_without_xi_list = []
ground_truth_ate_list = []

for _ in range(n_iterations):
    # Simulate data
    X = simulate_data(n_samples, d_cov, x_i_mean)
    weights = calculate_weights(d_cov)
    T, prob = calculate_propensity_scores(X, weights, x_i_selection_weight)
    Y, Y1, Y0 = simulate_outcomes(X, weights, T, treatment_effect, x_i_outcome_effect_weight)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled_with = scaler.fit_transform(X)
    X_scaled_without = scaler.fit_transform(X[:, 1:])
    
    # Fit logistic regression models for propensity scores
    propensity_model_with = LogisticRegression().fit(X_scaled_with, T)
    propensity_model_without = LogisticRegression().fit(X_scaled_without, T)
    
    propensity_scores_with = propensity_model_with.predict_proba(X_scaled_with)[:, 1]
    propensity_scores_without = propensity_model_without.predict_proba(X_scaled_without)[:, 1]
    
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
plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without)

# Display results
print(f"Averaged results over {n_iterations} iterations:")
print(f"Model: With x_i, ATE: {avg_ate_with_xi:.4f}, ATE Std: {std_ate_with_xi:.4f}")
print(f"Model: Without x_i, ATE: {avg_ate_without_xi:.4f}, ATE Std: {std_ate_without_xi:.4f}")
print(f"Ground Truth ATE: {avg_ground_truth_ate:.4f}, ATE Std: {std_ground_truth_ate:.4f}")

print(f"Average Absolute ATE Error (With x_i): {absolute_ate_error_with_xi:.4f}")
print(f"Average Absolute ATE Error (Without x_i): {absolute_ate_error_without_xi:.4f}")

# New code to plot absolute ATE error vs x_i_outcome_effect_weight

# Define a range of x_i_outcome_effect_weight values
x_i_outcome_effect_weights = [0, 0.01, 0.05, 0.1, 0.2,0.3,0.4, 0.5]

# Lists to store results for the new figure
absolute_ate_error_with_xi_list = []
absolute_ate_error_without_xi_list = []

# Loop over different x_i_outcome_effect_weights
for x_i_outcome_effect_weight in x_i_outcome_effect_weights:
    ate_with_xi_list = []
    ate_without_xi_list = []
    ground_truth_ate_list = []

    for _ in range(n_iterations):
        # Simulate data
        X = simulate_data(n_samples, d_cov, x_i_mean)
        weights = calculate_weights(d_cov)
        T, prob = calculate_propensity_scores(X, weights, x_i_selection_weight)
        Y, Y1, Y0 = simulate_outcomes(X, weights, T, treatment_effect, x_i_outcome_effect_weight)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled_with = scaler.fit_transform(X)
        X_scaled_without = scaler.fit_transform(X[:, 1:])
        
        # Fit logistic regression models for propensity scores
        propensity_model_with = LogisticRegression().fit(X_scaled_with, T)
        propensity_model_without = LogisticRegression().fit(X_scaled_without, T)
        
        propensity_scores_with = propensity_model_with.predict_proba(X_scaled_with)[:, 1]
        propensity_scores_without = propensity_model_without.predict_proba(X_scaled_without)[:, 1]
        
        # ATE estimation
        method = 'T-Learner'  # or 'IPW'
        ate_with_xi, ate_without_xi = calculate_ate(
            method, T, Y, X_scaled_with, X_scaled_without, propensity_scores_with, propensity_scores_without
        )
        
        # Store results for this iteration
        ate_with_xi_list.append(ate_with_xi)
        ate_without_xi_list.append(ate_without_xi)
        ground_truth_ate_list.append(np.mean(Y1 - Y0))

       # Calculate absolute ATE errors
    absolute_ate_error_with_xi = np.mean([abs(ate - gt) for ate, gt in zip(ate_with_xi_list, ground_truth_ate_list)])
    absolute_ate_error_without_xi = np.mean([abs(ate - gt) for ate, gt in zip(ate_without_xi_list, ground_truth_ate_list)])
    
    absolute_ate_error_with_xi_list.append(absolute_ate_error_with_xi)
    absolute_ate_error_without_xi_list.append(absolute_ate_error_without_xi)

# Plotting the results with additional annotations and compact figure
plt.figure(figsize=(7, 4))  # Reduced figure size for compactness
plt.plot(x_i_outcome_effect_weights, absolute_ate_error_with_xi_list, marker='o', label='Including X_i in Analysis')
plt.plot(x_i_outcome_effect_weights, absolute_ate_error_without_xi_list, marker='o', label='Excluding X_i from Analysis')

# Add vertical line at 0.2
plt.axvline(x=0.2, color='red', linestyle='--')

# Add annotations with adjusted position and font size
plt.text(0.1, max(absolute_ate_error_with_xi_list) * 1.4, 'Overlap violation', ha='center', color='red', fontsize=15)
plt.text(0.35, max(absolute_ate_error_with_xi_list) * 1.4, 'Ignorability violation', ha='center', color='red', fontsize=15)

# Updated labels and title to reflect the project context
plt.xlabel('X_i Outcome Effect Weight',fontsize=15)
plt.ylabel('Absolute ATE Error',fontsize=15)
#plt.title('Tradeoff Between Overlap and Ignorability Based on X_i Outcome Effect Weight', pad=0)  # Remove extra space before the title
plt.legend( fontsize=14, title_fontsize=10)  # Smaller legend for compactness
plt.grid(True)

# Adjust layout to remove unnecessary white space
plt.tight_layout()

# Show the plot
plt.show()

