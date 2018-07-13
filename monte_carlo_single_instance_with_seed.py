import sys
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from main_estimation import all_together, all_together_cross_fitting
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import joblib
import argparse
from joblib import delayed, Parallel

def experiment(x, eta, epsilon, tau, s_p, a_p, s_q, a_q, second_p, cube_p, lambda_reg):
     # Generate price as a function of co-variates
    p = np.dot(x[:, s_p], a_p) + eta
    # Generate demand as a function of price and co-variates
    q = tau * p + np.dot(x[:, s_q], a_q) + epsilon
    model_p = Lasso(alpha=lambda_reg) 
    model_q = Lasso(alpha=lambda_reg)
    return all_together_cross_fitting(x, p, q, second_p, cube_p, model_p=model_p, model_q=model_q)

def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=5000)
    parser.add_argument("--n_dim", dest="n_dim",
                        type=int, help='n_dim', default=1000)
    parser.add_argument("--n_experiments", dest="n_experiments",
                        type=int, help='n_experiments', default=2000)
    parser.add_argument("--support_size", dest="support_size",
                        type=int, help='support_size', default=400)
    parser.add_argument("--seed", dest="seed",
                        type=int, help='seed', default=12143)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)

    np.random.seed(opts.seed)
        
    '''
    We will work with a sparse linear model with high dimensional co-variates
    '''
    # Number of (price, demand, co-variate) samples
    n_samples = opts.n_samples
    # Dimension of co-variates
    n_dim = opts.n_dim
    # How many experiments to run to see the distribution of the recovered coefficient between price and demand
    n_experiments = opts.n_experiments
    support_size = opts.support_size 
    print("Support size of sparse functions: {}".format(support_size))

    '''
    True parameters
    '''

    # Support and coefficients for treatment as function of co-variates
    s_p = np.random.choice(range(n_dim), size=support_size, replace=False)
    a_p = np.random.uniform(0, 5, size=support_size)
    print("Support of treatment as function of co-variates: {}".format(s_p))
    print("Coefficients of treatment as function of co-variates: {}".format(a_p))

    # Distribution of residuals of treatment
    discounts = np.array([0, -.5, -2., -4.])
    probs = np.array([.65, .2, .1, .05])
    mean_discount = np.dot(discounts, probs)
    eta_sample = lambda x: np.array([discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])
    # Calculate moments of the residual distribution
    second_p = np.dot(probs, (discounts - mean_discount)**2)
    cube_p = np.dot(probs, (discounts - mean_discount)**3)
    quad_p = np.dot(probs, (discounts - mean_discount)**4)
    print("Second Moment of Eta: {:.2f}".format(second_p))
    print("Third Moment of Eta: {:.2f}".format(cube_p))
    print("Non-Gaussianity Criterion, E[eta^4] - 3 E[eta^2]^2: {:.2f}".format(quad_p - 3 * second_p**2))

    # Support and coefficients for outcome as function of co-variates
    s_q = s_p #np.random.choice(range(n_dim), size=support_size, replace=False)
    a_q = np.random.uniform(0, 5, size=support_size)
    print("Support of outcome as function of co-variates: {}".format(s_q))
    print("Coefficients of outcome as function of co-variates: {}".format(a_q))

    # Distribution of outcome residuals
    sigma_q = 1
    epsilon_sample = lambda x: np.random.uniform(-sigma_q, sigma_q, size=x)

    # Treatment effect
    tau = 3.0

    true_coef_p = np.zeros(n_dim)
    true_coef_p[s_p] = a_p
    true_coef_q = np.zeros(n_dim)
    true_coef_q[s_q] = a_q
    true_coef_q[s_p] += tau * a_p
    print(true_coef_q[s_q])
    '''
    Run  the experiments.
    '''
    # Coefficients recovered by orthogonal ML
    ortho_rec_tau = []
    first_stage_mse = []
    lambda_reg = np.sqrt(np.log(n_dim)/(n_samples))
    results = Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
                                            np.random.normal(size=(n_samples, n_dim)),
                                            eta_sample(n_samples),
                                            epsilon_sample(n_samples),
                                            tau, s_p, a_p, s_q, a_q, second_p, cube_p, lambda_reg
                                            ) for t in range(n_experiments))

    ortho_rec_tau = [[ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml] for ortho_ml, robust_ortho_ml, robust_ortho_est_ml, robust_ortho_est_split_ml, _, _ in results]
    first_stage_mse = [[np.linalg.norm(true_coef_p - coef_p), np.linalg.norm(true_coef_q - coef_q)] for _, _, _, _, coef_p, coef_q in results]

    print("Done with experiments!")

    joblib.dump(ortho_rec_tau, os.path.join(opts.output_dir,'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_seed_{}.jbl'.format(n_samples,n_dim,n_experiments,support_size, opts.seed)))
    joblib.dump(first_stage_mse, os.path.join(opts.output_dir,'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_seed_{}.jbl'.format(n_samples,n_dim,n_experiments,support_size, opts.seed)))


if __name__=="__main__":
    main(sys.argv[1:])