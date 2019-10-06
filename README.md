Conda environment:

    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p miniconda
    echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
    source ~/.bashrc


    for file_path in conda_envs/*.yml; do
        echo Using $file_path
        conda env create -f $file_path
    done


Run benchmark:

    log=singlethread_benchmark.log
    echo "======" > $log
    for file_path in conda_envs/*.yml; do
        env_name=$(basename $file_path .yml)
        conda activate $env_name
        echo $env_name >> $log
        python ./benchmark_scripts/conv2d_stack.py >> $log
        echo "======" >> $log 
        conda deactivate
    done

    log=multithread_benchmark.log
    echo "======" > $log
    for file_path in conda_envs/*.yml; do
        env_name=$(basename $file_path .yml)
        conda activate $env_name
        echo "environment:" $env_name >> $log
        python ./benchmark_scripts/conv2d_stack.py --num_threads=4 >> $log
        echo "======" >> $log 
        conda deactivate
    done
