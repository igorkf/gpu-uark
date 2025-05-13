#!/bin/bash

#SBATCH --job-name=datasets
#SBATCH --output=logs/create_datasets.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

## configs 
module purge
module load python
source myenv/bin/activate

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
mkdir -p data/Training_data
mkdir -p data/Testing_data
scp $SLURM_SUBMIT_DIR/data/Training_data/*.csv data/Training_data/
scp $SLURM_SUBMIT_DIR/data/Testing_data/*.csv data/Testing_data/
scp $SLURM_SUBMIT_DIR/output/geno_ok.csv output/
scp $SLURM_SUBMIT_DIR/src/*.py src/

#####################################################
## run tasks
#####################################################
fl=0 # filter locations? no (fl=0)
python3 -u src/create_datasets.py --fl=$fl > logs/create_datasets_fl_${fl}.log
#####################################################

## copy needed output files to /home
scp output/xtrain*.csv $SLURM_SUBMIT_DIR/output/
scp output/ytrain*.csv $SLURM_SUBMIT_DIR/output/
scp output/xval*.csv $SLURM_SUBMIT_DIR/output/
scp output/yval*.csv $SLURM_SUBMIT_DIR/output/
scp output/xtest*.csv $SLURM_SUBMIT_DIR/output/
scp logs/*.log $SLURM_SUBMIT_DIR/logs/
