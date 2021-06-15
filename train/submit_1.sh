#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=37

#SBATCH --mem=40G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti
#SBATCH --partition=standard
# SBATCH --partition=_workgroup_
# SBATCH --gres=gpu

#SBATCH --time=0-10:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --array=0-3

# export OMP_NUM_THREADS=4

vpkg_require xm_pytorch


start=$(date "+%s")
echo "Job Start: ${start}"
python3 -u data_generation.py
finish=$(date "+%s")
echo "Job Finish: ${finish}"
runtime=$(($finish-$start))
echo "Total Runtime: ${runtime}"
