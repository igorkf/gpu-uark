#!/bin/bash

#SBATCH --job-name=prep
#SBATCH --output=logs/preprocessing.out
#SBATCH --partition=condo
#SBATCH --constraint=samuelbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

## configs 
module purge
module load os/el9 gcc/14.2.1 mkl/21.3.0 R/4.4.1-el9-gnu

## copy needed files to /scratch
cd /scratch/$SLURM_JOB_ID
mkdir -p data src logs output
mkdir -p data/Training_data
mkdir -p data/Testing_data
scp $SLURM_SUBMIT_DIR/data/Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt data/Training_data
scp $SLURM_SUBMIT_DIR/data/Training_data/1_Training_Trait_Data_2014_2023.csv data/Training_data
scp $SLURM_SUBMIT_DIR/data/Testing_data/1_Submission_Template_2024.csv data/Testing_data
scp $SLURM_SUBMIT_DIR/src/prep_geno.R src/
scp $SLURM_SUBMIT_DIR/src/preprocessing.R src/

#####################################################
## run tasks
#####################################################
Rscript src/prep_geno.R > logs/prep_geno.log && \
    Rscript src/preprocessing.R > logs/preprocessing.log
#####################################################

## copy needed output files to /home
scp logs/prep_geno.log $SLURM_SUBMIT_DIR/logs/
scp logs/preprocessing.log $SLURM_SUBMIT_DIR/logs/
scp output/geno_ok.csv output/ $SLURM_SUBMIT_DIR/output/
scp output/train_val_test.csv $SLURM_SUBMIT_DIR/output/
