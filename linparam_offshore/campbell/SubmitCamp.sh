#!/bin/bash
#SBATCH --job-name=Camp                  # Job name
#SBATCH --time=04:00:00
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

cd IEA-15-RWT-Campbell/

../../../local/bin/openfast ws00.0.fst &
../../../local/bin/openfast ws03.0.fst &
../../../local/bin/openfast ws04.0.fst &
../../../local/bin/openfast ws05.0.fst &
../../../local/bin/openfast ws06.0.fst &
../../../local/bin/openfast ws07.0.fst &
../../../local/bin/openfast ws08.0.fst &
../../../local/bin/openfast ws09.0.fst &
../../../local/bin/openfast ws10.0.fst &
../../../local/bin/openfast ws11.0.fst &
../../../local/bin/openfast ws13.0.fst &
../../../local/bin/openfast ws14.0.fst &
../../../local/bin/openfast ws15.0.fst &
../../../local/bin/openfast ws17.5.fst &
../../../local/bin/openfast ws20.0.fst

wait
