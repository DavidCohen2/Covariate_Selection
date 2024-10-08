import numpy as np



# Helper functions
def simulate_data(n_samples, d_cov):
    X = np.random.normal(0, 1, (n_samples, d_cov))
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


def create_data_generate_process(mode='mode_1', X=None, T=None, weights=None, x_i_outcome_effect_weight=None):
    
    if mode=='mode_1':
        # Parameters
        n_samples = 1000  # Number of samples 
        d_cov = 10  # Total number of covariates including x_i
        treatment_effect = 3

        x_i_selection_weight = 10
        x_i_outcome_effect_weight = 0.3
        
        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        z = np.random.normal(0, 1, d_cov - 1)
        weights = z / np.sqrt(d_cov - 1)
        
        # Calculate propensity scores
        l = np.dot(X[:, 1:], weights) + x_i_selection_weight * X[:, 0]
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)
        
        # Simulate outcomes
        Y0 = np.dot(X[:, 1:], weights) + x_i_outcome_effect_weight * X[:, 0]**2
        Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob

    if mode == 'mode_folds_step1':
        # Parameters
        n_samples = 1000  # Number of samples
        d_cov = 10  # Total number of covariates including x_i

        x_i_selection_weight = 10
        x_i_outcome_effect_weight = 0.3

        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        z = np.random.normal(0, 1, d_cov - 1)
        weights = z / np.sqrt(d_cov - 1)

        # Calculate propensity scores
        l = np.dot(X[:, 1:], weights) + x_i_selection_weight * X[:, 0]
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)

        return X, T, prob, weights

    if mode == 'mode_folds_step2':
        treatment_effect = 3
        # Simulate outcomes
        Y0 = np.dot(X[:, 1:], weights) + x_i_outcome_effect_weight * X[:, 0] ** 2 + np.sin(X[:, 1] ** 2)*X[:, 2]**2 + np.sin(X[:, 1] * X[:, 2])
        Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return Y, Y1, Y0


