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
module load python/2024.02-1-anaconda
python3 --version

## create python environment
python -m venv myenv # create an virtual environment called "myenv"
source myenv/bin/activate
pip install --upgrade pip

## install ligthgbm (it needs GCC 9 to compile)
module load gcc/9
pip install cmake ninja lightgbm
module purge gcc/9

# install other libs
nvidia-smi # check CUDA version to see if GPU is working
pip install -r requirements.txt # install packages
pip list # list packages
