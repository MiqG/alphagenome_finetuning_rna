#!/usr/bin/env bash
#SBATCH --account=ehpc708
#SBATCH --job-name=snakepipe
#SBATCH --no-requeue
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --gres=none
#SBATCH --partition=gpp
#SBATCH --qos=gp_ehpc
#SBATCH --output=snakepipe.%x_%A.out
#SBATCH --error=snakepipe.%x_%A.err

set -eo pipefail

# unpack arguments
SNAKEMAKE_COMMAND=$1

# run command
eval $SNAKEMAKE_COMMAND

echo "Done!"