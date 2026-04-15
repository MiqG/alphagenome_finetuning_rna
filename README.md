# AlphaGenome Fine-tuning on SF3B1-Mutant RNA-seq

Snakemake pipeline for fine-tuning AlphaGenome-PyTorch and Borzoi on SF3B1-mutant MEC1 RNA-seq data.

## Running the Full Pipeline

### Local execution
```bash
snakemake --use-conda -j <cores>
```

### SLURM cluster submission
```bash
sbatch src/scripts/submit_snakemake_slurm.sh 'snakemake --cluster "sbatch --cpus-per-task={threads} --mem={resources.memory}G --time={resources.runtime} --partition={resources.partition} --gres={resources.gres} --parsable" --cluster-status src/scripts/status-sacct.sh --jobs 25 --use-conda -s workflows/Snakefile --latency-wait 60 --keep-going --rerun-incomplete'
```

### Dry run
```bash
snakemake -n
```

---

## Testing Fine-tuning with Overfitting

Before running the full pipeline, test the fine-tuning workflow on a small set of intervals using the **overfitting workflow**. This quickly verifies that:
- The training loop is functioning correctly
- The model can converge on a tiny dataset
- Predictions have reasonable magnitudes (not NaN/inf)
- Data loading works for all modalities (RNA-seq + splice junctions)

### Quick Start (10-15 min)

**Step 1: Run overfitting on 16 intervals**
```bash
snakemake -s workflows/overfit_alphagenome.smk overfit_sf3b1mut --use-conda -j 1
```

This will:
- Extract 16 intervals from `data/prep/finetuning/alphagenome/FOLD_0/train.bed` (~5 sec)
- Fine-tune for 100 epochs with constant learning rate (no warmup/decay) (~10-15 min on V100 GPU)
- Converge to loss → 0 within 20-50 epochs
- Save checkpoint to `results/finetuning/alphagenome_pytorch/overfit/overfit/best_model.pth`

**Timing:**
- Total runtime: 10-15 min on GPU

**Step 2: Visualize predictions**
```bash
snakemake -s workflows/overfit_alphagenome.smk visualize_overfit --use-conda -j 1
```

This will:
- Load the overfitting checkpoint
- Run inference on each of the 16 training intervals
- Generate a 6-panel PDF per interval comparing real vs predicted tracks
- Save to `results/finetuning/alphagenome_pytorch/overfit/visualization/tracks.pdf`

**Step 3: Inspect results**
```bash
# Check training convergence
tail results/finetuning/alphagenome_pytorch/overfit/overfit/training_log.csv

# View visualization
xdg-open results/finetuning/alphagenome_pytorch/overfit/visualization/tracks.pdf
```

### Overfitting Hyperparameters

The overfitting rule differs from the full pipeline to enable fast convergence on 16 intervals:

| Parameter | Full Pipeline | Overfitting | Reason |
|-----------|---------------|-------------|--------|
| `epochs` | 10 | 100 | Converge on tiny dataset |
| `train_bed` / `val_bed` | FOLD_0/{train,valid}.bed | Same 16-interval BED | Check both use training data |
| `warmup_steps` | 200 | 0 | No warmup masking on tiny data |
| `lr_schedule` | cosine | constant | Constant LR for clearer signal |
| `gradient_accumulation_steps` | 8 | 1 | Immediate updates on 16 samples |
| `gradient_checkpointing` | enabled | enabled | Same as pipeline |

### Expected Results

#### Training Convergence
The training loss should converge to ~0 within 20-50 epochs:
```
Epoch  1 - Train Loss: 2.345 - Val Loss: 2.301
Epoch 10 - Train Loss: 0.234 - Val Loss: 0.239
Epoch 30 - Train Loss: 0.001 - Val Loss: 0.002
Epoch 50 - Train Loss: 0.0001 - Val Loss: 0.0001
```

If loss does **not** decrease:
- Check that GPU is being used (`nvidia-smi`)
- Verify learning rate is not too low (should be ~1e-4)
- Check that model weights are actually being updated (inspect checkpoint file size)

#### Visualization PDF
The PDF should show:
- **Row 0-1 (RNA-seq)**: Predicted signal closely follows real signal (especially after convergence)
- **Row 2-3 (Splice classification)**: Predictions peak at positions of real splice junctions
- **Row 4-5 (Splice usage)**: Predicted usage pattern matches real multi-sample usage

Red flags:
- Predicted signal is all zeros → model not learning
- Predicted signal is constant across positions → model overfitting to global mean
- Predictions are NaN/inf → numerical instability (gradient explosion or bad data)

### Troubleshooting

**Missing junction files**
```
Error: Missing input file data/raw/ENA/sf3b1mut/STAR/SRR17111301/second_pass.SJ.fwd.tab
```
Some samples may not have strand-split junction files. Check what files exist:
```bash
ls data/raw/ENA/sf3b1mut/STAR/SRR17111301/
```

If only `second_pass.SJ.out.tab` exists (no `.fwd.tab` / `.rev.tab`), edit `workflows/overfit_alphagenome.smk` line ~73 to adjust `OVERFIT_SAMPLES` to samples that have split junction files.

**Out of memory**
If running on limited GPU memory:
```bash
# Edit workflows/overfit_alphagenome.smk and reduce:
sequence_length: 131072  # instead of 1048576
```

**Visualization script dependencies**
If the visualization fails with missing imports, install dependencies:
```bash
pip install pyfaidx pyBigWig matplotlib
```

### Full Workflow

To run the complete pipeline:
1. Verify overfitting works (as above)
2. Run the full pipeline:
   ```bash
   snakemake --use-conda -j 4
   ```
3. Monitor with:
   ```bash
   snakemake --use-conda -j 4 --dryrun
   snakemake --use-conda -j 4 --dag | dot -Tpng > dag.png
   ```

---

## Configuration

All pipeline parameters are centralized in `config/config.yaml`. See `CLAUDE.md` for detailed architecture and configuration reference.
