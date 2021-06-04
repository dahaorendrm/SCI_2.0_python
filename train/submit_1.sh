#!/bin/bash -l

#BATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=40

#SBATCH --mem=80G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti
#SBATCH --partition=standard
# SBATCH --partition=_workgroup_
# SBATCH --gres=gpu

#SBATCH --time=1-1:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=201-232


vpkg_require xm_pytorch
python3 data_generation.py

