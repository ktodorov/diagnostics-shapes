#!/bin/bash
#SBATCH --job-name=joint_best
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:1
module purge
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

for disabled_property in 0 1 2 3 4
do
    for alpha in 0.25 0.5 0.75
    do
        srun python3 -u train_game.py --device cuda --seed 114 --disabled-properties $disabled_property --multi-task --multi-task-lambda $alpha --iterations 30000 --lr 0.1 >> 'output/multi-disabled-'$disabled_property'-alpha-'$alpha'-seed-114.out'
    done
done