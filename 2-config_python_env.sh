#!/bin/bash

#SBATCH --job-name=env
#SBATCH --output=logs/config_python_env.out
#SBATCH --partition=gpu06 # this partition has GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

## configs 
module purge
module load python

## create python environment
echo "Creating Python environment..."
python3 -m venv gpu-uark # create virtual environment
source gpu-uark/bin/activate
pip3 install --upgrade pip
nvidia-smi # check CUDA version
pip3 install -r requirements.txt
pip3 list
