#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J HNL_PU40_train
#SBATCH --mail-user=daniel.bb0321@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 8:30:00
#SBATCH -A m3443

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
file_ind=1
config="train_configs/test.yaml"
module load conda
conda activate trackml

python3 make_configs.py $config $file_ind
traintrack configs/pipeline_$file_ind.yaml
python3 read_param.py $config

cd ../analysis/HSF
python3 roc/plt_performance.py $file_ind
python3 tracks/DBSCAN_GridSearch.py $file_ind
