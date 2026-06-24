#!/usr/bin/env bash
#SBATCH --job-name=snakepipe
#SBATCH --no-requeue
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=genoa64
#SBATCH --qos=pipelines
#SBATCH --output=snakepipe.%x_%A.out
#SBATCH --error=snakepipe.%x_%A.err

set -eo pipefail

# unpack arguments
SNAKEMAKE_COMMAND=$1

# run command
eval $SNAKEMAKE_COMMAND

echo "Done!"