import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso
from sklearn.cross_validation import KFold

def all_together(x, p, q, second_p, cube_p,
                 model_p=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]),
                 model_q=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100])):
    # Split the data in half, train and test
    t_split = x.shape[0] // 2
    x_train, p_train, q_train = x[:t_split], p[:t_split], q[:t_split]
    x_test, p_test, q_test  = x[t_split:], p[t_split:], q[t_split:]
    
    # Fit with LassoCV the treatment as a function of x and the outcome as
    # a function of x, using only the train fold
    model_p.fit(x_train, p_train)
    model_q.fit(x_train, q_train)
    
    # Then compute residuals p-g(x) and q-q(x) on test fold
    res_p = (p_test - model_p.predict(x_test)).flatten()
    res_q = (q_test - model_q.predict(x_test)).flatten()
    
    ''' ORTHO ML '''
    # Compute coefficient by OLS on residuals
    ortho_ml = np.sum(np.multiply(res_p, res_q))/np.sum(np.multiply(res_p, res_p))
    
    ''' ROBUST ORTHO ML with KNOWN MOMENTS '''
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p 
    #
    # The coefficient is a simple division: 
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    mult_p = res_p**3 - 3 * second_p * res_p - cube_p
    robust_ortho_ml = np.mean(res_q * mult_p)/np.mean(res_p * mult_p)
    
    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS '''
    # Estimate the moments from the residuals of the first fold
    res_p_first = p_train - model_p.predict(x_train)
    second_p_est = np.mean(res_p_first**2)
    cube_p_est = np.mean(res_p_first**3) - 3 * np.mean(res_p_first) * np.mean(res_p_first**2)
    # Estimate the treatment effect from the second fold
    mult_p_est = res_p**3 - 3 * second_p_est * res_p - cube_p_est
    robust_ortho_est_ml = np.mean(res_q * mult_p_est)/np.mean(res_p * mult_p_est)
    
    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT '''
    # Estimate the moments from the residuals of the first fold
    test_split = x_test.shape[0] // 2
    res_p_first = res_p[:test_split]
    second_p_est = np.mean(res_p_first**2)
    cube_p_est = np.mean(res_p_first**3) - 3 * np.mean(res_p_first) * np.mean(res_p_first**2)
    # Estimate the treatment effect from the second fold
    res_p_second = res_p[test_split:]
    res_q_second = res_q[test_split:]
    mult_p_est = res_p_second**3 - 3 * second_p_est * res_p_second - cube_p_est
    robust_ortho_est_split_ml = np.mean(res_q_second * mult_p_est)/np.mean(res_p_second * mult_p_est)
    
    
    return ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml



def all_together_cross_fitting(x, p, q, second_p, cube_p,
                 model_p=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100]),
                 model_q=LassoCV(alphas=[0.01, 0.1, 0.3, 0.5, 0.9, 5, 10, 20, 100])):
    
    res_p = np.zeros(x.shape[0])
    res_q = np.zeros(x.shape[0])
    mult_p = np.zeros(x.shape[0])
    mult_p_est = np.zeros(x.shape[0])
    mult_p_est_split = np.zeros(x.shape[0])
    
    kf = KFold(x.shape[0], n_folds=2)
    for train_index, test_index in kf:
        # Split the data in half, train and test
        x_train, p_train, q_train = x[train_index], p[train_index], q[train_index]
        x_test, p_test, q_test  = x[test_index], p[test_index], q[test_index]
        
        # Fit with LassoCV the treatment as a function of x and the outcome as
        # a function of x, using only the train fold
        model_p.fit(x_train, p_train)
        model_q.fit(x_train, q_train)
        
        # Then compute residuals p-g(x) and q-q(x) on test fold
        res_p[test_index] = (p_test - model_p.predict(x_test)).flatten()
        res_q[test_index] = (q_test - model_q.predict(x_test)).flatten()

        # Estimate multipliers for robust orthogonal methods 
        
        # 1. Multiplier with known moments
        mult_p[test_index] = res_p[test_index]**3 - 3 * second_p * res_p[test_index] - cube_p 
        
        # 2. Multiplier with estimated moments on training data
        res_p_first = p_train - model_p.predict(x_train)
        second_p_est = np.mean(res_p_first**2)
        cube_p_est = np.mean(res_p_first**3) - 3 * np.mean(res_p_first) * np.mean(res_p_first**2)
        # Estimate the treatment effect from the second fold
        mult_p_est[test_index] = res_p[test_index]**3 - 3 * second_p_est * res_p[test_index] - cube_p_est
        
        # 3. Multiplier with estimated moments on further split and cross-fit of test data
        nested_kf = KFold(len(test_index), n_folds=2)
        for nested_train_index, nested_test_index in nested_kf:
            res_p_first = res_p[test_index[nested_train_index]]
            second_p_est = np.mean(res_p_first**2)
            cube_p_est = np.mean(res_p_first**3) - 3 * np.mean(res_p_first) * np.mean(res_p_first**2)
            res_p_second = res_p[test_index[nested_test_index]]
            mult_p_est_split[test_index[nested_test_index]] = res_p_second**3 - 3 * second_p_est * res_p_second - cube_p_est

    ''' ORTHO ML '''
    # Compute coefficient by OLS on residuals
    ortho_ml = np.mean(res_q * res_p)/np.mean(res_p * res_p)
    
    ''' ROBUST ORTHO ML with KNOWN MOMENTS '''
    # Compute for each sample the quantity:
    #
    #          (Z_i-f(X_i))^3 - 3*(sigma^2)*(Z_i-f(X_i)) - cube_p 
    #
    # The coefficient is a simple division: 
    #
    #       E_n{ (Y-m(X)) * ((Z-f(X))^3-3(sigma^2)(Z-f(X))) }
    #   -----------------------------------------------------------------
    #   E_n{ (Z-f(x)) * ((Z-f(x))^3 - 3 * (sigma^2) * (Z-f(x)) - cube_p)}
    #
    robust_ortho_ml = np.mean(res_q * mult_p)/np.mean(res_p * mult_p)
    
    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS '''
    robust_ortho_est_ml = np.mean(res_q * mult_p_est)/np.mean(res_p * mult_p_est)
    
    ''' ROBUST ORTHO ML with ESTIMATED MOMENTS on THIRD SPLIT '''
    robust_ortho_est_split_ml = np.mean(res_q * mult_p_est_split) / np.mean(res_p * mult_p_est_split)
    
    
    return ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, model_p.coef_, model_q.coef_
