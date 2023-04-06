#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=6

#SBATCH --mem=60G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=lesti2.0_allstep
# SBATCH --partition=_workgroup_
# SBATCH --partition=standard
# SBATCH --partition=gpu-t4
#SBATCH --partition=idle
#SBATCH --gpus=1

#SBATCH --time=0-4:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/20210806-LESTI_2.0_DAIN
python3 -u sim_test.py
