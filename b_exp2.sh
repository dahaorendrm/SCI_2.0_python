#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=8

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=lesti_S0_exp
#SBATCH --partition=gpu-t4
#SBATCH --gpus=1

#SBATCH --time=0-20:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/20210806-LESTI_2.0_DAIN
python3 -u exp_test.py
