# AlphaGenome Overfitting & Visualization Workflow

## Overview

This standalone Snakefile (`workflows/overfit_alphagenome.smk`) provides debugging tools for AlphaGenome fine-tuning:

1. **`rule create_overfit_bed`** — Extract first 16 intervals from FOLD_0 for quick overfitting
2. **`rule overfit_sf3b1mut`** — Fine-tune on those 16 intervals in ~100 optimizer steps
3. **`rule visualize_overfit`** — Generate a multi-page PDF comparing real vs predicted tracks

This is independent of the main pipeline and useful for:
- Verifying the training loop is working correctly
- Checking model predictions are sensible (not NaN, proper scale)
- Inspecting splice junction predictions
- Debugging data loading issues

---

## Files Created

| File | Purpose |
|------|---------|
| `workflows/overfit_alphagenome.smk` | Standalone Snakefile with 3 rules |
| `src/alphagenome-pytorch/scripts/visualize_overfit.py` | Inference + matplotlib plotting |
| `OVERFIT_WORKFLOW.md` | This documentation |

---

## Quick Start

### 1. Run overfitting (CPU/GPU, ~5-10 min)

```bash
snakemake -s workflows/overfit_alphagenome.smk overfit_sf3b1mut --use-conda
```

**What it does:**
- Creates `support/overfit.bed` with 16 intervals
- Calls `torchrun` to fine-tune on those 16 intervals
- Saves `results/finetuning/alphagenome_pytorch/overfit/overfit/best_model.pth`

**Hyperparameters** (optimized for fast overfitting):
- `--epochs 100` (enough to see convergence on 16 samples)
- `--lr-schedule constant`, `--warmup-steps 0` (no learning rate decay)
- `--batch-size 1`, `--gradient-accumulation-steps 1` (immediate updates)
- `--save-every-steps 50` (checkpoints every 50 steps)

**Training should converge to ~0 loss** after 20-50 epochs if the code is working correctly.

### 2. Visualize predictions (CPU, ~2-5 min)

```bash
snakemake -s workflows/overfit_alpiagenome.smk visualize_overfit --use-conda
```

**What it does:**
- Loads the overfitting checkpoint
- Runs inference on each of the 16 training intervals
- Creates a 6-panel PDF per interval:
  - **Row 0-1:** Real vs predicted RNA-seq signal
  - **Row 2-3:** Real vs predicted splice site classification
  - **Row 4-5:** Real vs predicted splice site usage
- Saves `results/finetuning/alphagenome_pytorch/overfit/visualization/tracks.pdf`

### 3. Run both rules

```bash
snakemake -s workflows/overfit_alphagenome.smk --use-conda
```

---

## Interpreting Results

### Training log

Check convergence in the training log:

```bash
ls results/finetuning/alphagenome_pytorch/overfit/overfit/
# training_log.csv (if saved by TrainingLogger)
# best_model.pth (final checkpoint)
# checkpoint_epoch*.pth (intermediate)
```

**Expected behavior:**
- Train loss decreases rapidly (to ~0 after 20-50 epochs)
- Val loss mirrors train loss (no overfitting signal, as train == val data)

### Visualization PDF

Open the PDF with any PDF viewer:

```bash
open results/finetuning/alphagenome_pytorch/overfit/visualization/tracks.pdf
# or
xdg-open results/finetuning/alphagenome_pytorch/overfit/visualization/tracks.pdf
```

**Expected patterns:**
- **Predicted RNA-seq**: Should follow real signal closely (especially after convergence)
- **Splice sites**: Should show predictions at real splice junction positions
- **Splice usage**: Should match pattern of multi-sample usage (if training on multiple samples)

**Red flags:**
- Predicted signal is all zeros → model not learning
- Predicted signal is all NaNs → numerical instability
- Predictions don't move over epochs → learning rate too low, or model not training

---

## Configuration

The overfitting rule inherits most config from the main pipeline:

```yaml
# From config/config.yaml
finetuning:
  alphagenome:
    sf3b1mut:
      sequence_length: 1048576
      overlap_highres: 1024
      lr: 1e-4  # (inherited)
      track_means_samples: 500
```

**Overrides in the Snakefile:**
- `epochs: 100` (not 10)
- `lr_schedule: constant` (not cosine)
- `warmup_steps: 0` (not 200)
- `gradient_accumulation_steps: 1` (not 8)
- `save_every_steps: 50` (not 500)

To change these, edit `workflows/overfit_alphagenome.smk` lines ~90-94.

---

## Troubleshooting

### Missing STAR junction files

If the rule fails with missing `.SJ.fwd.tab` / `.SJ.rev.tab`:
- Some samples may not have strand-split junction files
- Check `data/raw/ENA/sf3b1mut/STAR/{sample}/` for what files exist
- The rule uses `OVERFIT_SAMPLES = SAMPLES[:2]` (first 2 samples); verify they have required files

### GPU out of memory

- Reduce `--sequence-length` in the Snakefile (e.g., 131072 instead of 1048576)
- Set `--gradient-checkpointing` to False if memory is tight

### Visualization script fails

Check that all dependencies are installed in the alphagenome_pytorch conda env:

```bash
pip install pyfaidx pyBigWig matplotlib seaborn
```

---

## Files Generated

### Training artifacts

```
results/finetuning/alphagenome_pytorch/overfit/
├── overfit/
│   ├── best_model.pth              # Final checkpoint (loaded by visualize)
│   ├── checkpoint_epoch_0.pth      # Intermediate checkpoints (every 50 steps)
│   ├── checkpoint_epoch_1.pth
│   └── .done                        # Touch file (rule completed)
└── visualization/
    └── tracks.pdf                  # Multi-page PDF (one page per interval)
```

### Data files

```
support/
└── overfit.bed                      # 16-interval BED file (created by rule)
```

---

## Notes

- The overfitting rule uses **sequence parallelism** (same as the main pipeline) — this distributes the long sequence across GPUs, not the batch
- The visualization script uses plain matplotlib (no JAX dependency) for simplicity
- Both rules use the same conda environment (`alphagenome_pytorch.yaml`)
- Run time: ~5-10 min training + ~2-5 min visualization on a single V100/A100 GPU
