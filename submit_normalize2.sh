#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_train_leakyrelu
# SBATCH --partition=standard
#SBATCH --partition=_workgroup_
#SBATCH --gres=gpu

#SBATCH --time=2-10:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch
python3 -u train-normalize2.py

