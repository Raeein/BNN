#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes to use
#SBATCH --job-name=TE_32_16
#SBATCH --output=z_%j.out
#SBATCH --error=z_%j.err
#SBATCH --time=24:00:00                # Maximum time for the job 
#SBATCH --mem=64G                      # Memory required per node 
#SBATCH --cpus-per-task=4              # Number of CPUs 
#SBATCH --mail-user=USERNAME@uoguelph.ca # Your email
#SBATCH --mail-type=ALL                # Get email for all job events

# Setup the python env
module load python/3.10.13

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install the deps
pip install --no-index -r requirements.txt

export PYTHONUNBUFFERED=1

python -u bnn_elastic_net_five_folds.py
