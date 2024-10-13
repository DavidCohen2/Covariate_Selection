import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples and treatment effect
n_samples = 500
treatment_effect = 2.0

# Generate feature X[:, 0] (well-overlapping feature)
X0 = np.random.uniform(0, 10, n_samples)

# Generate feature X[:, 1] (feature with overlap problem)
X1_control = np.random.uniform(0, 5, n_samples // 2)  # Control group with X1 in [0, 5]
X1_treatment = np.random.uniform(5, 10, n_samples // 2)  # Treatment group with X1 in [5, 10]

# Combine X1 and treatment assignment
X1 = np.concatenate([X1_control, X1_treatment])
X = np.column_stack((X0, X1))

# Treatment assignment (0 for control, 1 for treatment)
T = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)])

# Outcome under control
Y0 = 5 + 0.5 * X[:, 0] + 0.2 * np.sin(X[:, 1])

# Outcome under treatment
Y1 = Y0 + treatment_effect

# Observed outcome
Y = Y0 * (1 - T) + Y1 * T

# Split data into training and testing sets
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.2, random_state=42
)

# T-Learner with X[:, 0] only (ignoring X[:, 1])
X_train_T_only_X0 = X_train[:, [0]]  # Using only X[:, 0]
X_test_T_only_X0 = X_test[:, [0]]

# T-Learner with X[:, 0] and X[:, 1]
X_train_T_X0_X1 = X_train  # Using both X[:, 0] and X[:, 1]
X_test_T_X0_X1 = X_test

# Train separate models for treatment and control
# Using only X[:, 0]
model_control_X0 = LinearRegression().fit(
    X_train_T_only_X0[T_train == 0], Y_train[T_train == 0]
)
model_treatment_X0 = LinearRegression().fit(
    X_train_T_only_X0[T_train == 1], Y_train[T_train == 1]
)

# Using X[:, 0] and X[:, 1]
model_control_X0_X1 = LinearRegression().fit(
    X_train_T_X0_X1[T_train == 0], Y_train[T_train == 0]
)
model_treatment_X0_X1 = LinearRegression().fit(
    X_train_T_X0_X1[T_train == 1], Y_train[T_train == 1]
)

# Predict Y0 and Y1 using T-Learner with X[:, 0] only
Y0_pred_X0 = model_control_X0.predict(X_test_T_only_X0)
Y1_pred_X0 = model_treatment_X0.predict(X_test_T_only_X0)

# Predict Y0 and Y1 using T-Learner with X[:, 0] and X[:, 1]
Y0_pred_X0_X1 = model_control_X0_X1.predict(X_test_T_X0_X1)
Y1_pred_X0_X1 = model_treatment_X0_X1.predict(X_test_T_X0_X1)

# Estimate treatment effect
treatment_effect_pred_X0 = Y1_pred_X0 - Y0_pred_X0
treatment_effect_pred_X0_X1 = Y1_pred_X0_X1 - Y0_pred_X0_X1

# True treatment effect
true_treatment_effect = treatment_effect * np.ones_like(Y_test)

# Calculate ATE for both models
ATE_pred_X0 = np.mean(treatment_effect_pred_X0)
ATE_pred_X0_X1 = np.mean(treatment_effect_pred_X0_X1)
ATE_true = treatment_effect

# Calculate Absolute Error for ATE
ATE_error_X0 = np.abs(ATE_pred_X0 - ATE_true)
ATE_error_X0_X1 = np.abs(ATE_pred_X0_X1 - ATE_true)

# Print ATE results
print("Average Treatment Effect (ATE) Estimates:")
print(f"True ATE: {ATE_true}")
print(f"ATE using only X[:, 0]: {ATE_pred_X0:.4f} (Error: {ATE_error_X0:.4f})")
print(f"ATE using X[:, 0] and X[:, 1]: {ATE_pred_X0_X1:.4f} (Error: {ATE_error_X0_X1:.4f})")

# # Plot the results
# plt.figure(figsize=(14, 6))
#
# # Subplot 1: T-Learner with X[:, 0] only
# plt.subplot(1, 2, 1)
# plt.scatter(X_test[:, 0], treatment_effect_pred_X0, color='blue', alpha=0.6, label='Predicted ATE')
# plt.axhline(y=ATE_true, color='black', linestyle='--', label='True ATE')
# plt.xlabel('X[:, 0]')
# plt.ylabel('Estimated Treatment Effect')
# plt.title('T-Learner using only X[:, 0]')
# plt.legend()
#
# # Subplot 2: T-Learner with X[:, 0] and X[:, 1]
# plt.subplot(1, 2, 2)
# plt.scatter(X_test[:, 0], treatment_effect_pred_X0_X1, color='red', alpha=0.6, label='Predicted ATE')
# plt.axhline(y=ATE_true, color='black', linestyle='--', label='True ATE')
# plt.xlabel('X[:, 0]')
# plt.ylabel('Estimated Treatment Effect')
# plt.title('T-Learner using X[:, 0] and X[:, 1]')
# plt.legend()
#
# plt.suptitle('Comparison of ATE Estimation with and without Overlap Issue')
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

