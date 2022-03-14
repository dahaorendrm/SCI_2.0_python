#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --mem=60G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_S0_S3traindata
# SBATCH --partition=_workgroup_
#SBATCH --partition=idle
# SBATCH --partition=devel
# SBATCH --gres=gpu

#SBATCH --time=0-10:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/20210806-LESTI_2.0_DAIN
#cd S0_gaptv
python3 -u S0_run.py
#cd ..
