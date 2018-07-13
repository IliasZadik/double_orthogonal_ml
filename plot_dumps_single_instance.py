import sys
import os
import argparse
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--input_dir", dest="input_dir", type=str, default=".")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)

    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    
    for n_experiments in [2000]:
        for n_samples in [2000, 5000, 10000]:  
            for n_dim in [1000, 2000, 5000]:
                for sigma_q in [3, 10, 20]:         
                    if n_samples==10000 and n_dim > 1000:
                        continue
                    if n_samples==5000 and n_dim > 2000:
                        continue
                    print(n_samples, n_dim)
                    bias = []
                    std = []
                    first_mse = []
                    if n_samples == 2000:
                        supports = [1, 20, 40, 60, 80, 100, 150]
                    if n_samples == 5000:
                        supports = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
                    if n_samples == 10000:
                        supports = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800]
                        
                    for support_size in supports:
                        if os.path.exists(os.path.join(opts.input_dir,'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}'.format(n_samples,n_dim,n_experiments,support_size,sigma_q))):
                            ortho_rec_tau = joblib.load(os.path.join(opts.input_dir,'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}'.format(n_samples,n_dim,n_experiments,support_size,sigma_q)))
                            first_stage_mse = joblib.load(os.path.join(opts.input_dir,'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}'.format(n_samples,n_dim,n_experiments,support_size,sigma_q)))
                            bias.append(np.mean(ortho_rec_tau, axis=0) - 3)
                            std.append(np.std(ortho_rec_tau, axis=0))
                            first_mse.append(np.mean(first_stage_mse, axis=0))
                    plt.figure(figsize=(15, 5))
                    plt.suptitle("Second Stage Estimate Distributions.\n n_samples: {}, n_dim: {}".format(n_samples, n_dim))
                    plt.subplot(1,4,1)
                    plt.title('bias')
                    plt.plot(supports, np.array(bias)[:, [0,3]])
                    plt.xlabel('support size')
                    plt.legend(['dml', 'second_order'])
                    plt.subplot(1,4,2)
                    plt.title('std')
                    plt.plot(supports, np.array(std)[:, [0,3]])
                    plt.xlabel('support size')
                    plt.legend(['dml', 'second_order'])
                    plt.subplot(1,4,3)
                    plt.title('mse')
                    plt.plot(supports, (np.array(std)**2 + np.array(bias)**2)[:, [0,3]])
                    plt.xlabel('support size')
                    plt.legend(['dml', 'second_order'])
                    plt.subplot(1,4,4)
                    plt.title("L2-norm Errors of First Stage Models.\n n_samples: {}, n_dim: {}".format(n_samples, n_dim))
                    plt.plot(supports, np.array(first_mse)[:, 0], label='model_p')
                    plt.plot(supports, np.array(first_mse)[:, 1], label='model_q')
                    plt.xlabel('support size')
                    plt.legend()
                    plt.tight_layout(pad=5)
                    plt.savefig(os.path.join(opts.output_dir,'comparison_of_each_method_n_samples_{}_n_dim_{}_n_exp_{}_sigma_q_{}.png'.format(n_samples,n_dim,n_experiments,sigma_q)), dpi=300)
                    plt.savefig(os.path.join(opts.output_dir,'comparison_of_each_method_n_samples_{}_n_dim_{}_n_exp_{}_sigma_q_{}.pdf'.format(n_samples,n_dim,n_experiments,sigma_q)), dpi=300)

if __name__=="__main__":
    main(sys.argv[1:])
