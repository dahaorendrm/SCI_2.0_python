#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=10G
# SBATCH --mem-per-cpu=10G

#SBATCH --job-name=build
# SBATCH --partition=_workgroup_
#SBATCH --partition=standard
#SBATCH --partition=devel
#SBATCH --gres=gpu


vpkg_require xm_pytorch/SCI_2.0_t4
# !/usr/bin/env bash

echo "Need pytorch>=1.0.0"
#source activate pytorch1.0.0

export PYTHONPATH=$PYTHONPATH:$(pwd)

cd MinDepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConv
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd InterpolationCh
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd DepthFlowProjection
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd Interpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd SeparableConvFlow
rm -rf build *.egg-info dist
python setup.py install
cd ..

cd FilterInterpolation
rm -rf build *.egg-info dist
python setup.py install
cd ..

