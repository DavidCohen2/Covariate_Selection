import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression


def estimate_ate_ipw(T, Y, propensity_score):
    weights = T / propensity_score + (1 - T) / (1 - propensity_score)
    weighted_outcome = weights * Y
    ate = np.mean(weighted_outcome[T == 1]) - np.mean(weighted_outcome[T == 0])
    return ate

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