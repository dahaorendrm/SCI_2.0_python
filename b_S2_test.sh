#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=20G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=chasti_S2_test
# SBATCH --partition=_workgroup_
#SBATCH --partition=standard
# SBATCH --partition=devel
#SBATCH --gres=gpu

#SBATCH --time=0-01:00:00
# SBATCH --output=ArraySCI%A-%a.out
#SBATCH --mail-user='xmdrm@udel.edu'
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
# SBATCH --array=4-10

# export OMP_NUM_THREADS=4
vpkg_require xm_pytorch/SCI_2.0_t4
cd S2_flow_predict
python3 -u test.py
cd ..
mkdir -p S3_spectra_convert/data/test
cp -r S2_denoiser/result/*.npz S3_spectra_convert/data/test
nvidia-smi
