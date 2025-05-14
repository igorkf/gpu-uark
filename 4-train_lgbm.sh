#!/bin/bash

#SBATCH --job-name=lgbm
#SBATCH --output=logs/train_lgbm.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

## configs 
module purge
module load python
source myenv/bin/activate

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
scp $SLURM_SUBMIT_DIR/output/x*.csv output/
scp $SLURM_SUBMIT_DIR/output/y*.csv output/
scp $SLURM_SUBMIT_DIR/src/*.py src/

#####################################################
## run tasks
#####################################################
python -u src/train_lgbm.py > logs/train_lgbm.log
#####################################################

## copy needed output files to /home
scp output/pred_lgbm.csv $SLURM_SUBMIT_DIR/output/
scp output/importance_lgbm.csv $SLURM_SUBMIT_DIR/output/
scp logs/train_lgbm.log $SLURM_SUBMIT_DIR/logs/
