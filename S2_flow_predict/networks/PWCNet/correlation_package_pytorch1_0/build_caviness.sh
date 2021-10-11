#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=10G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=build_PWC
# SBATCH --partition=_workgroup_
#SBATCH --partition=standard
# SBATCH --partition=devel
#SBATCH --gres=gpu:t4

# !/usr/bin/env bash

vpkg_require xm_pytorch/SCI_2.0_t4

echo "Need pytorch>=1.0.0"
#source activate pytorch1.0.0

export PYTHONPATH=$PYTHONPATH:$(pwd)/../../my_package

rm -rf build *.egg-info dist
python setup.py install
