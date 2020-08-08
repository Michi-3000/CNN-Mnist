#!/bin/bash
#SBATCH -J myresnet
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6 
#SBATCH --gres=gpu:1
#SBATCH --output=myresnet.out
#SBATCH -t 10:00:00

module load pytorch-py37-cuda10.1-gcc/1.4.0 

python myresnet.py --num_gpus=1 --batch_size=50 --variable_update=parameter_serve