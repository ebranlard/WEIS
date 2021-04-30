#!/bin/bash
#SBATCH --job-name=Twr                  # Job name
#SBATCH --time=47:00:00
#SBATCH --nodes=1                               # Number of nodes
#SBATCH --ntasks-per-node=36                    # Number of processors per node
#SBATCH -A bar                           # Allocation
#SBATCH --mail-user emmanuel.branlard@nrel.gov  # E-mail adres
#SBATCH --mail-type BEGIN,END,FAIL              # Send e-mail when job begins, ends or fails
#SBATCH -o slurm-%x-%j.log                      # Output

echo "Modules..."
module purge
module load conda
module load comp-intel intel-mpi mkl


echo "Conda..."
source activate
conda activate weis-env

echo "Run.."
python 1_ParamTwr.py
