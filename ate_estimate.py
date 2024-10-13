import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from xgboost import XGBRegressor
#from catboost import CatBoostRegressor

def estimate_ate_ipw(T, Y, propensity_score):
    weights = T / propensity_score + (1 - T) / (1 - propensity_score)
    weighted_outcome = weights * Y
    ate = np.mean(weighted_outcome[T == 1]) - np.mean(weighted_outcome[T == 0])
    return ate

def estimate_ate_t_learner(T, Y, X, estimator):
    model_treated = estimator.__class__()  # Create a new instance of the estimator
    model_control = estimator.__class__()  # Create another new instance of the estimator
    
    model_treated.fit(X[T == 1], Y[T == 1])
    model_control.fit(X[T == 0], Y[T == 0])
    
    y_pred_treated = model_treated.predict(X)
    y_pred_control = model_control.predict(X)
    ate = np.mean(y_pred_treated - y_pred_control)
    return ate

def estimate_ate_polynomial_t_learner(T, Y, X, degree):
    model_treated = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model_control = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    
    model_treated.fit(X[T == 1], Y[T == 1])
    model_control.fit(X[T == 0], Y[T == 0])
    
    y_pred_treated = model_treated.predict(X)
    y_pred_control = model_control.predict(X)
    ate = np.mean(y_pred_treated - y_pred_control)
    return ate

def get_estimator(estimator_name, degree=None):
    if estimator_name == 'LinearRegression':
        return LinearRegression()
    elif estimator_name == 'PolynomialRegression2':
        return make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    elif estimator_name == 'PolynomialRegression3':
        return make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    elif estimator_name == 'CatBoost':
        return CatBoostRegressor(verbose=0)
    elif estimator_name == 'XGBoost':
        return XGBRegressor()
    elif estimator_name == 'GradientBoosting':
        return GradientBoostingRegressor()
    else:
        raise ValueError(f"Unknown estimator: {estimator_name}")

def calculate_ate(method, T, Y, X_scaled_with, X_scaled_without, propensity_scores_with, propensity_scores_without, estimator_name=None):
    estimator = get_estimator(estimator_name)
    if method == 'IPW':
        ate_with_xi = estimate_ate_ipw(T, Y, propensity_scores_with)
        ate_without_xi = estimate_ate_ipw(T, Y, propensity_scores_without)
    elif method == 'T-Learner':
        if 'Polynomial' in estimator_name:
            degree = int(estimator_name[-1])
            ate_with_xi = estimate_ate_polynomial_t_learner(T, Y, X_scaled_with, degree)
            ate_without_xi = estimate_ate_polynomial_t_learner(T, Y, X_scaled_without, degree)
        else:
            ate_with_xi = estimate_ate_t_learner(T, Y, X_scaled_with, estimator)
            ate_without_xi = estimate_ate_t_learner(T, Y, X_scaled_without, estimator)
    else:
        raise ValueError("Method must be either 'IPW' or 'T-Learner'")
    return ate_with_xi, ate_without_xi