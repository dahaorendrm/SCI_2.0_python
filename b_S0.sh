#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=60G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_S0_S3traindata
# SBATCH --partition=_workgroup_
#SBATCH --partition=standard
# SBATCH --partition=devel
# SBATCH --gres=gpu

#SBATCH --time=3-10:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/SCI_2.0_t4
cd S0_gaptv
python3 -u run_gap_tv.py
cd ..
