#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_test
# SBATCH --partition=standard
# SBATCH --partition=_workgroup_
#SBATCH --partition=devel
#SBATCH --gres=gpu

#SBATCH --time=0-2:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch
python3 -u S1_test_meaN.py

