#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 4
#SBATCH -q regular
#SBATCH -J HNL_PU40_train
#SBATCH --mail-user=daniel.bb0321@gmail.com
#SBATCH --mail-type=ALL
#SBATCH -t 16:30:00
#SBATCH -A m3443

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

#run the application:
#applications may perform better with --gpu-bind=none instead of --gpu-bind=single:1 
module load conda
conda activate trackml

srun --ntasks-per-node 1 -c 32 --cpu_bind=cores -G 4 --gpu-bind=single:1 traintrack configs/noPU_pipeline_1.yaml

