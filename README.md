# Introduction

Code associated with paper: <a href="https://arxiv.org/abs/1711.00342">Orthogonal Machine Learning: Power and Limitations</a>, Mackey, Syrgkanis, Zadik, ICML 2018

# File descriptions

* `main_estimation.py` : contains the implementation of all the proposed second order orthogonal methods and all the benchmark first order orthogonal estimation methods for the partially linear model

* `monte_carlo_single_instance_with_seed.py` and `monte_carlo_single_instance.py` : almost identical files that generate data from the partially linear model DGP and based on input parameters, and run the proposed second order orthogonal method and all the benchmarks and save the results in joblib dumps. The only difference is in the naming convention of the result files. The first is used in the generation of the plots with multiple instances and the second in the generation of the plots with a single instance and as we sweep over parameters. 

* `plot_dumps_multi_instance.py` and `plot_dumps_single_instance.py` : plot the figures in the paper from the corresponding dumpes in the files above. 

* `single_instance_parameter_sweep.sh` : shell script that runs a single instance as the parameters of the DGP vary and creates all the related plots

* `multi_instance.sh` : shell script that runs multiple instances of the DGP for a fixed set of parameters and generated the related plots

# Re-creating the Figures in the Paper

To recreate the figures in the paper run the following script:
```bash
./single_instance_parameter_sweep.sh single_instance_dir
./multi_instance_parameter_sweep.sh multi_instance_dir
```

This will create the figures in the relative folder: ./figures

These scripts take a very long time and it is advisable that they run on a cluster. For our results we parallelized the for loops in each of these shell scripts and run it on multiple nodes on a cluster.