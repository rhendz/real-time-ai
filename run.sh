#!/bin/bash
#SBATCH --job-name rai-stl
#SBATCH -N 1
#SBATCH -o jobs/rai/rai.out.%A
#SBATCH -e jobs/rai/rai.err.%A
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=48
#SBATCH --ntasks-per-node=1

echo Running on $SLURM_JOB_NUM_NODES nodes
date
srun python3 predictcifar10.py
date

