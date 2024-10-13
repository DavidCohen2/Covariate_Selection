import numpy as np
from sklearn.preprocessing import PolynomialFeatures


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

        x_i_selection_weight = 1
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
    
    if mode=='mode_2d':
        # Parameters
        n_samples = 150  # Number of samples 
        d_cov = 5  # Total number of covariates including x_i
        treatment_effect = 3

        overlap_violation_factor = 20
        x_i_selection_weight = 3
        x_i_outcome_effect_weight = 0.1
        
        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        z = np.random.normal(0, 1, d_cov - 1)
        weights = z / np.sqrt(d_cov - 1)
        
        # Calculate propensity scores
        l = overlap_violation_factor*np.dot(X[:, 1:], weights) + x_i_selection_weight * X[:, 0]
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)
        
        # Simulate outcomes
        Y0 = np.dot(X[:, 1:], weights) + x_i_outcome_effect_weight * X[:, 0]
        Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob

    if mode == 'relu':
        # Parameters
        n_samples = 200  # Number of samples
        d_cov = 2  # Total number of covariates including x_i
        treatment_effect = 3

        x_i_selection_weight = 50
        x_i_outcome_effect_weight = 0.2

        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        X[:, 1] = np.sign(X[:, 0]) * np.abs(X[:, 1])

        # if X[:, 1] > 0, then T = 1 else T = 0
        #l = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_selection_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        # prob equal 1 for         
        T = (X[:, 1] > 0).astype(int)
        prob = T
        
        # Simulate outcomes
        # Y0 relu of X[:, 0] and linear function of X[:, 1]
        Y0 = np.maximum(0, X[:, 0]) + 2 * X[:, 1]        
        Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob
    
    if mode == 'mode_poly_all':
        # Parameters
        n_samples = 200  # Number of samples
        d_cov = 3  # Total number of covariates including x_i
        treatment_effect = 3

        x_i_selection_weight = 50
        x_i_outcome_effect_weight = 0.2

        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Separate terms involving X[:,0]
        terms_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] > 0]
        terms_not_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] == 0]
        
        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))
 
        # Calculate propensity scores
        l = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_selection_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)

        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))
        
        # Simulate outcomes
        Y0 = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_outcome_effect_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob

    if mode == 'mode_poly_all2':
        # Parameters
        n_samples = 100  # Number of samples
        d_cov = 20  # Total number of covariates including x_i
        treatment_effect = 3

        x_i_selection_weight = 50
        x_i_outcome_effect_weight = 0.2

        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Separate terms involving X[:,0]
        terms_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] > 0]
        terms_not_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] == 0]
        
        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))

        # Calculate propensity scores
        l = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_selection_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        l = np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)*30
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)

        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))
        
        
        # Simulate outcomes
        z = np.random.normal(0, 1, d_cov - 1)
        weights = z / np.sqrt(d_cov - 1)

        Y0 = np.dot(X[:, 1:]**2, weights) + x_i_outcome_effect_weight * X[:, 0]
        Y1 = Y0 + treatment_effect


        #Y0 = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_outcome_effect_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        #Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob

    if mode == 'mode_poly_all3':
        # Parameters
        n_samples = 100  # Number of samples
        d_cov = 2  # Total number of covariates including x_i
        treatment_effect = 3

        x_i_selection_weight = 10
        x_i_outcome_effect_weight = 0.2

        # Simulate data
        X = np.random.normal(0, 1, (n_samples, d_cov))
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # Separate terms involving X[:,0]
        terms_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] > 0]
        terms_not_involving_x0 = [i for i, term in enumerate(poly.powers_) if term[0] == 0]
        
        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))

        # Calculate propensity scores
        l = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_selection_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        l = np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)*30


        l = -(X[:, 1]+X[:, 0])*10
        prob = 1 / (1 + np.exp(-l))
        T = np.random.binomial(1, prob)

        # Normalize weights
        z = np.random.normal(0, 1,len(terms_involving_x0))
        weights_involving_x0 = z / np.sqrt(len(terms_involving_x0))

        z = np.random.normal(0, 1,len(terms_not_involving_x0))
        weights_not_involving_x0 = z / np.sqrt(len(terms_not_involving_x0))
        
        
        # Simulate outcomes
        z = np.random.normal(0, 1, d_cov - 1)
        weights = z / np.sqrt(d_cov - 1)

        Y0 = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0)
        Y0 = X[:, 1]*X[:, 0]
        Y1 = Y0 + treatment_effect


        #Y0 = np.dot(X_poly[:, terms_not_involving_x0], weights_not_involving_x0) + x_i_outcome_effect_weight * np.dot(X_poly[:, terms_involving_x0], weights_involving_x0)
        #Y1 = Y0 + treatment_effect

        Y = T * Y1 + (1 - T) * Y0

        return X, T, Y, Y1, Y0, prob

    
    

