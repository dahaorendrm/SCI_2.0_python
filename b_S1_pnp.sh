#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=70G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_S1_train
#SBATCH --partition=gpu-v100
#SBATCH --gpus=1
# SBATCH --gres=gpu:t4:1  
#SBATCH --time=1-00:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/20210806-LESTI_2.0_DAIN
python3 -u S1_pnp.py

