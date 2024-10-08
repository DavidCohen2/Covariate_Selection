
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


    
