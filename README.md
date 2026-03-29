
```{shell}
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --cpus-per-task={threads} --mem={resources.memory}G --time={resources.runtime} --partition={resources.partition} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 25 --use-conda -s workflows/Snakefile --latency-wait 60 --keep-going --rerun-incomplete'
```
