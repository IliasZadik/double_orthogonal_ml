dir=$1
mkdir -p $dir
echo $dir

for n_samples in 5000; 
do
    for n_experiments in 2000;
    do
        for n_dim in 1000;
        do
            for support_size in 1 20 40 60 80 100 150 200 250 300 350 400 450 500 550;
            do
                for instance in $(seq 1 100);
                do
                    python3 monte_carlo_single_instance_with_seed.py --n_samples $n_samples --n_experiments $n_experiments --n_dim $n_dim --support_size $support_size --output_dir $dir --seed $instance
                done
            done
        done
    done
done

mkdir -p figures
mkdir -p figures/multi_instance
python3 plot_multi_instance.py --input_dir $dir --output_dir figures/multi_instance

echo "Saved figures in figures/multi_instance!" 