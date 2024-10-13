
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



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
    for i, (scores, label) in enumerate(zip([propensity_scores_with, propensity_scores_without], ['Including x_i', 'Excluding x_i'])):
        plt.subplot(1, 2, i+1)
        plt.hist(scores[T == 1], bins=30, alpha=0.5, label=f'Treated ({label})')
        plt.hist(scores[T == 0], bins=30, alpha=0.5, label=f'Control ({label})')
        plt.title(f'Propensity Score Distribution ({label})')
        plt.xlabel('Propensity Score')
        plt.ylabel('Frequency')
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ate_error_vs_x_i_outcome_effect_weight(x_i_outcome_effect_weights, mean_ate_error_with_xi_list, mean_ate_error_without_xi_list, std_ate_error_with_xi_list, std_ate_error_without_xi_list):
    # Plotting the results with additional annotations and compact figure
    plt.figure(figsize=(7, 4))  # Reduced figure size for compactness

    # Plot with error bars
    plt.errorbar(x_i_outcome_effect_weights, mean_ate_error_with_xi_list,
                 yerr=std_ate_error_with_xi_list, marker='o', label='Including X_i in Analysis', capsize=5)
    plt.errorbar(x_i_outcome_effect_weights, mean_ate_error_without_xi_list,
                 yerr=std_ate_error_without_xi_list, marker='o', label='Excluding X_i from Analysis', capsize=5)

    # Add vertical line at 0.2
    plt.axvline(x=0.2, color='red', linestyle='--')

    # Add annotations with adjusted position and font size
    plt.text(0.1, max(mean_ate_error_with_xi_list) * 1.4, 'Overlap violation', ha='center', color='red', fontsize=15)
    plt.text(0.35, max(mean_ate_error_with_xi_list) * 1.4, 'Ignorability violation', ha='center', color='red',
             fontsize=15)

    # Updated labels and title to reflect the project context
    plt.xlabel('X_i Outcome Effect Weight', fontsize=15)
    plt.ylabel('Absolute ATE Error', fontsize=15)
    # plt.title('Tradeoff Between Overlap and Ignorability Based on X_i Outcome Effect Weight', pad=0)
    plt.legend(fontsize=14, title_fontsize=10)  # Smaller legend for compactness
    plt.grid(True)

    # Adjust layout to remove unnecessary white space
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_ates_and_cohen_d_vs_x_i_outcome_effect_weight(x_i_outcome_effect_weights, mean_ate_with_xi_list, mean_ate_without_xi_list, cohen_d_list, std_ate_with_xi_list, std_ate_without_xi_list):
    # Plotting the results with additional annotations and compact figure
    plt.figure(figsize=(7, 4))  # Reduced figure size for compactness

    # Plot ATE with error bars
    plt.errorbar(x_i_outcome_effect_weights, mean_ate_with_xi_list,
                 yerr=std_ate_with_xi_list, marker='o', label='ATE with X_i', capsize=5)
    plt.errorbar(x_i_outcome_effect_weights, mean_ate_without_xi_list,
                 yerr=std_ate_without_xi_list, marker='o', label='ATE without X_i', capsize=5)

    # Plot Cohen's d without error bars
    plt.plot(x_i_outcome_effect_weights, cohen_d_list, marker='o', label='Cohen d')

    # Updated labels and title to reflect the project context
    plt.xlabel('X_i Outcome Effect Weight', fontsize=15)
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


    
