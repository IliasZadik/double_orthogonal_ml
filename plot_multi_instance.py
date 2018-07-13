import sys
import os
import argparse
import glob
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

    multi_instance_dump_dir = opts.input_dir

    for n_experiments in [2000]:
        for n_samples in [5000]:        
            for n_dim in [1000]:
                if n_samples==10000 and n_dim > 1000:
                    continue
                if n_samples==5000 and n_dim > 2000:
                    continue
                bias = []
                std = []
                bias_gr = []
                std_gr = []
                first_gr = []
                first_mse = []
                if n_samples == 2000:
                    supports = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
                if n_samples == 5000:
                    supports = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
                if n_samples == 10000:
                    supports = [1, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550]
                    
                for s, support_size in enumerate(supports):
                    for it, file in enumerate(glob.glob(os.path.join(multi_instance_dump_dir,'recovered_coefficients_from_each_method_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_seed_*'.format(n_samples,n_dim,n_experiments,support_size)))):
                        if it == 0:
                            bias.append([])
                            std.append([])
                            bias_gr.append(support_size)
                            std_gr.append(support_size)
                        ortho_rec_tau = joblib.load(file)
                        bias[s].append(np.mean(ortho_rec_tau, axis=0) - 3)
                        std[s].append(np.std(ortho_rec_tau, axis=0))
                    for it, file in enumerate(glob.glob(os.path.join(multi_instance_dump_dir, 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_seed_*'.format(n_samples,n_dim,n_experiments,support_size)))):
                        if it == 0:                        
                            first_mse.append([])
                            first_gr.append(support_size)
                        first_stage_mse = joblib.load(file)
                        first_mse[s].append(np.mean(first_stage_mse, axis=0))


    def plot_estimates(plot_bias_gr, bias, method_inds, methods):
        for i, m in zip(method_inds,methods):
            median_bias = [np.median(np.array(bias[s])[:, i]) for s in range(len(plot_bias_gr))]
            max_bias = [np.max(np.array(bias[s])[:, i]) for s in range(len(plot_bias_gr))]
            min_bias = [np.min(np.array(bias[s])[:, i]) for s in range(len(plot_bias_gr))]
            plt.plot(plot_bias_gr, median_bias, '*', label=m)            
            plt.fill_between(plot_bias_gr, min_bias, max_bias, alpha=0.5)

    print([np.array(bias[s]).shape[0] for s in range(len(bias_gr))])
    plot_bias_gr = bias_gr
    methods = ['dml', '2nd order']
    method_inds = [0,3]

    plt.figure(figsize=(15, 5))
    plt.suptitle("Second Stage Estimate Distributions.\n n_samples: {}, n_dim: {}".format(n_samples, n_dim))
    plt.subplot(1,4,1)
    plt.title('BIAS')
    plot_estimates(plot_bias_gr, bias, method_inds, methods)
    plt.xlabel('support size')
    plt.legend()
    plt.subplot(1,4,2)
    plt.title("STD")
    plot_estimates(plot_bias_gr, std, method_inds, methods)
    plt.xlabel('support size')
    plt.legend()
    plt.subplot(1,4,3)
    plt.title("MSE")
    for i, m in zip([0,3], methods):
        median_bias = [np.median((np.array(bias[s])**2 + np.array(std[s])**2)[:, i]) for s in range(len(plot_bias_gr))]
        max_bias = [np.max((np.array(bias[s])**2 + np.array(std[s])**2)[:, i]) for s in range(len(plot_bias_gr))]
        min_bias = [np.min((np.array(bias[s])**2 + np.array(std[s])**2)[:, i]) for s in range(len(plot_bias_gr))]
        plt.plot(plot_bias_gr, median_bias, '*', label=m)            
        plt.fill_between(plot_bias_gr, min_bias, max_bias, alpha=0.5)
    plt.xlabel('support size')
    plt.legend()

    plt.subplot(1,4,4)
    plt.title("First Stage $\ell_2$ error")
    plot_estimates(plot_bias_gr, first_mse, [0,1], ['model_p', 'model_q'])
    plt.xlabel('support size')
    plt.legend()
    plt.tight_layout(pad=5)
    plt.savefig(os.path.join(opts.output_dir,'multi_instance_comparison_of_each_method_n_samples_{}_n_dim_{}_n_exp_{}.png'.format(n_samples,n_dim,n_experiments)), dpi=300)
    plt.savefig(os.path.join(opts.output_dir,'multi_instance_comparison_of_each_method_n_samples_{}_n_dim_{}_n_exp_{}.pdf'.format(n_samples,n_dim,n_experiments)), dpi=300)

if __name__=="__main__":
    main(sys.argv[1:])