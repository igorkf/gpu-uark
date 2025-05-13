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
python3 -m venv myenv # create an virtual environment called "myenv"
source myenv/bin/activate # activate it
pip3 install --upgrade pip # upgrade installation manager (pip)
nvidia-smi # check CUDA version
pip3 install -r requirements.txt # install packages
pip3 list # list packages
