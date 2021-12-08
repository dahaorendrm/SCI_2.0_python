#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_S1_train
#SBATCH --partition=gpu-t4
#SBATCH --gpus=1
# SBATCH --gres=gpu:t4:1  
#SBATCH --time=1-10:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/SCI_2.0_t4
cd S3_spectra_convert
python3 -u train.py
cd ..
