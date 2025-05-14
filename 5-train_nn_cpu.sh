#!/bin/bash

#SBATCH --job-name=nn_cpu
#SBATCH --output=logs/train_nn_cpu.out
#SBATCH --partition=cloud72
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00

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
loss=mse
model=G
bs=128
gamma=0.6
fl=0
device=cpu
python3 -u src/train_nn.py --model=$model --loss=$loss --bs=$bs --gamma=$gamma --fl=$fl --device=$device > logs/train_nn_${model}_${loss}_${bs}_${gamma}_fl_${fl}_device_${device}.log
#####################################################

## copy needed output files to /home
scp output/pred_nn_${model}_${loss}_${bs}_${gamma}_fl_${fl}_device_${device}.csv $SLURM_SUBMIT_DIR/output/
scp logs/train_nn_${model}_${loss}_${bs}_${gamma}_fl_${fl}_device_${device}.log $SLURM_SUBMIT_DIR/logs/
