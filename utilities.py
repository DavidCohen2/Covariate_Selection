
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def estimate_propensity_score(X, T):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression model for propensity scores
    propensity_model = LogisticRegression().fit(X_scaled, T)
    propensity_scores = propensity_model.predict_proba(X_scaled)[:, 1]
    
    return propensity_scores

def plot_propensity_score_distribution(propensity_scores_with, propensity_scores_without, T):
    plt.figure(figsize=(14, 6))
    for i, (scores, label) in enumerate(zip([propensity_scores_with, propensity_scores_without], ['Including $x_{co}$', 'Excluding $x_{co}$'])):
        plt.subplot(1, 2, i+1)
        plt.hist(scores[T == 1], bins=30, alpha=0.5, label=f'Treated ({label})')
        plt.hist(scores[T == 0], bins=30, alpha=0.5, label=f'Control ({label})')
        plt.title(f'Propensity Score Distribution ({label})')
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_ate_error_vs_x_i_outcome_effect_weight(x_i_outcome_effect_weights, mean_ate_error_with_xi_list, mean_ate_error_without_xi_list):
    # Plotting the results with additional annotations and compact figure
    plt.figure(figsize=(7, 4))  # Reduced figure size for compactness
    plt.plot(x_i_outcome_effect_weights, mean_ate_error_with_xi_list, marker='o',
             label='Including $x_{co}$ in Analysis')
    plt.plot(x_i_outcome_effect_weights, mean_ate_error_without_xi_list, marker='o',
             label='Excluding $x_{co}$ from Analysis')

    # Add vertical line at 0.2
    plt.axvline(x=0.1, color='red', linestyle='--')

    # Add annotations with adjusted position and font size
    plt.text(0.03, max(mean_ate_error_with_xi_list) * 1.4, 'Overlap\nviolation', ha='center', color='red', fontsize=12)
    plt.text(0.4, max(mean_ate_error_with_xi_list) * 1.4, 'Ignorability violation', ha='center', color='red',
             fontsize=12)

    # Updated labels and title to reflect the project context
    plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel('Absolute ATE Error', fontsize=15)
    # plt.title('Tradeoff Between Overlap and Ignorability Based on X_i Outcome Effect Weight', pad=0)  # Remove extra space before the title
    plt.legend(fontsize=14, title_fontsize=10)  # Smaller legend for compactness
    plt.grid(True)

    # Adjust layout to remove unnecessary white space
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_ates_and_cohen_d_vs_x_i_outcome_effect_weight(x_i_outcome_effect_weights, mean_ate_with_xi_list, mean_ate_without_xi_list, cohen_d_list):
    # Plotting the results with additional annotations and compact figure
    plt.figure(figsize=(7, 4))  # Reduced figure size for compactness
    plt.plot(x_i_outcome_effect_weights, mean_ate_with_xi_list, marker='o', label='ATE with $x_{co}$')
    plt.plot(x_i_outcome_effect_weights, mean_ate_without_xi_list, marker='o',
             label='ATE without $x_{co}$')
    plt.plot(x_i_outcome_effect_weights, cohen_d_list, marker='o',
             label='Cohen\'s d')

    # Add vertical line at 0.2
    #plt.axvline(x=0.2, color='red', linestyle='--')

    # Add annotations with adjusted position and font size
    # plt.text(0.1, max(mean_ate_error_with_xi_list) * 1.4, 'Overlap violation', ha='center', color='red', fontsize=15)
    # plt.text(0.35, max(mean_ate_error_with_xi_list) * 1.4, 'Ignorability violation', ha='center', color='red',
    #          fontsize=15)

    # Updated labels and title to reflect the project context
    plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel('ATE', fontsize=15)
    plt.legend(fontsize=14, title_fontsize=10)  # Smaller legend for compactness
    plt.grid(True)

    # Adjust layout to remove unnecessary white space
    plt.tight_layout()

    # Show the plot
    plt.show()


def calculate_ate_statistics(ate_list, ground_truth_ate_list, n_iterations):
    # Calculate average ATE and standard deviation over all iterations
    avg_ate = np.mean(ate_list)
    std_ate = np.std(ate_list)
    sem_ate = std_ate / np.sqrt(n_iterations)

    ate_error = np.mean([abs(ate - gt) for ate, gt in zip(ate_list, ground_truth_ate_list)])
    sem_ate_error = np.std([abs(ate - gt) for ate, gt in zip(ate_list, ground_truth_ate_list)]) / np.sqrt(n_iterations)

    return {
        "avg_ate": avg_ate,
        "sem_ate": sem_ate,
        "ate_error": ate_error,
        "sem_ate_error": sem_ate_error
    }



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def polynomial_t_learner(T, Y, X, degree=2):
    model_treated = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model_control = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    
    model_treated.fit(X[T == 1], Y[T == 1])
    model_control.fit(X[T == 0], Y[T == 0])
    
    #y_pred_treated = model_treated.predict(X)
    #y_pred_control = model_control.predict(X)
    
    return model_treated,model_control

def plot_data(X, T, Y0, Y1, Y0_without_noise, Y1_without_noise, X_grid, Y0_grid, Y1_grid):
    # Plotting T
    plt.figure(figsize=(10, 6))
    plt.scatter(X[T == 1, 1], X[T == 1, 0], color='green', label='Treated (T=1)')
    plt.scatter(X[T == 0, 1], X[T == 0, 0], color='blue', label='Control (T=0)')
    plt.xlabel('X[:, 1]')
    plt.ylabel('X[:, 0]')
    plt.legend()
    plt.title('Treated vs Control')
    plt.show()

    # # Plotting Y0
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 1], Y0, color='red', label='Y0')
    # plt.xlabel('X[:, 1]')
    # plt.ylabel('Y0')
    # plt.legend()
    # plt.title('Y0 vs X[:, 1]')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 0], Y0, color='red', label='Y0')
    # plt.xlabel('X[:, 0]')
    # plt.ylabel('Y0')
    # plt.legend()
    # plt.title('Y0 vs X[:, 0]')
    # plt.show()

    # # Plotting Y1
    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 1], Y1, color='purple', label='Y1')
    # plt.xlabel('X[:, 1]')
    # plt.ylabel('Y1')
    # plt.legend()
    # plt.title('Y1 vs X[:, 1]')
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.scatter(X[:, 0], Y1, color='purple', label='Y1')
    # plt.xlabel('X[:, 0]')
    # plt.ylabel('Y1')
    # plt.legend()
    # plt.title('Y1 vs X[:, 0]')
    # plt.show()

    # # 3D Plotting Y0_without_noise
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], Y0_without_noise, color='red', label='Y0_without_noise')
    # ax.set_xlabel('X[:, 0]')
    # ax.set_ylabel('X[:, 1]')
    # ax.set_zlabel('Y0_without_noise')
    # ax.legend()
    # ax.set_title('Y0_without_noise vs X[:, 0] and X[:, 1]')
    # plt.show()

    # 3D Plotting Y1_without_noise
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], Y1_without_noise, color='purple', label='Y1_without_noise')
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y1_without_noise')
    ax.legend()
    ax.set_title('Y1_without_noise vs X[:, 0] and X[:, 1]')
    plt.show()

    # 3D Plotting Y0_grid
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y0_grid.reshape(50, 50), color='red', alpha=0.5)
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y0_grid')
    ax.set_title('Y0_grid vs X[:, 0] and X[:, 1]')
    plt.show()

    # 3D Plotting Y1_grid
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y1_grid.reshape(50, 50), color='purple', alpha=0.5)
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y1_grid')
    ax.set_title('Y1_grid vs X[:, 0] and X[:, 1]')
    plt.show()

    # 3D Plotting Y0_grid and Y1_grid on the same figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y0_grid.reshape(50, 50), color='red', alpha=0.5, label='Y0_grid')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y1_grid.reshape(50, 50), color='purple', alpha=0.5, label='Y1_grid')
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y0_grid and Y1_grid')
    ax.set_title('Y0_grid and Y1_grid vs X[:, 0] and X[:, 1]')
    ax.legend()
    plt.show()


    Y = T * Y1 + (1 - T) * Y0
    model_treated,model_control =  polynomial_t_learner(T, Y, X, degree=2)
    Y1_grid_pred = model_treated.predict(X_grid)
    Y0_grid_pred = model_control.predict(X_grid)


    # 3D Plotting Y0_grid with Y0(T=0) points
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y0_grid.reshape(50, 50), color='blue', alpha=0.5, label='Y0_grid')
    # Add predicted Y0 points surface
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y0_grid_pred.reshape(50, 50), color='cyan', alpha=0.5, label='Y0_grid_pred')
    ax.scatter(X[T == 0, 0], X[T == 0, 1], Y0[T == 0], color='blue', label='Y0 (Control)')
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y0')
    ax.set_title('Y0_grid vs X[:, 0] and X[:, 1]')
    ax.legend()
    plt.show()

    # 3D Plotting Y1_grid with Y1(T=1) points
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y1_grid.reshape(50, 50), color='green', alpha=0.5, label='Y1_grid')
    # Add predicted Y1 points surface
    ax.plot_surface(X_grid[:, 0].reshape(50, 50), X_grid[:, 1].reshape(50, 50), Y1_grid_pred.reshape(50, 50), color='lightgreen', alpha=0.5, label='Y1_grid_pred')
    ax.scatter(X[T == 1, 0], X[T == 1, 1], Y1[T == 1], color='green', label='Y1 (Treated)')
    ax.set_xlabel('X[:, 0]')
    ax.set_ylabel('X[:, 1]')
    ax.set_zlabel('Y1')
    ax.set_title('Y1_grid vs X[:, 0] and X[:, 1]')
    ax.legend()
    plt.show()

    model_treated, model_control = polynomial_t_learner(T, Y, X[:, 1].reshape(-1, 1), degree=2)
    Y1_grid_pred = model_treated.predict(X_grid[:, 1].reshape(-1, 1))
    Y0_grid_pred = model_control.predict(X_grid[:, 1].reshape(-1, 1))


    # 2D plot of Y0_grid with Y0(T=0) points for X[:, 1]
    plt.figure(figsize=(10, 6))    
    plt.scatter(X_grid[:, 1], Y0_grid, color='blue', label='Y0_grid')
    plt.plot(X_grid[:, 1], Y0_grid_pred, color='cyan', label='Y0_grid_pred')
    plt.scatter(X[T == 0, 1], Y0[T == 0], color='blue', label='Y0 (Control)')
    plt.xlabel('X[:, 1]')
    plt.ylabel('Y0')
    plt.legend()
    plt.title('Y0_grid vs X[:, 1]')
    plt.show()

    # 2D plot of Y1_grid with Y1(T=1) points for X[:, 1]
    plt.figure(figsize=(10, 6))
    plt.scatter(X_grid[:, 1], Y1_grid, color='green', label='Y1_grid')
    plt.plot(X_grid[:, 1], Y1_grid_pred, color='lightgreen', label='Y1_grid_pred')
    plt.scatter(X[T == 1, 1], Y1[T == 1], color='green', label='Y1 (Treated)')
    plt.xlabel('X[:, 1]')
    plt.ylabel('Y1')
    plt.legend()
    plt.title('Y1_grid vs X[:, 1]')
    plt.show()

    


    


    
    
