# Plan: track wall time and GPU peak memory per epoch

## Context
README step 3 (`original` group) requires storing "time and memory" per epoch.
Currently neither is captured: finetune.py has profiling support for the first
N batches but nothing writes wall time or GPU memory to `epoch_log.csv`.

## Changes

### 1. `finetune.py` — wrap the train + val block with timing and memory reset

At the top of the epoch loop (just before the `train_epoch_*` call), add:

```python
import time
epoch_t0 = time.perf_counter()
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)
```

After the validation call (just before `logger.log_epoch`), add:

```python
epoch_wall_time = time.perf_counter() - epoch_t0
peak_mem_gb = (
    torch.cuda.max_memory_allocated(device) / 1e9
    if torch.cuda.is_available() else 0.0
)
extra["epoch_wall_time_s"] = epoch_wall_time
extra["peak_gpu_mem_gb"] = peak_mem_gb
```

`extra` is already built just before `logger.log_epoch(...)` at ~line 1857,
so these two keys flow into `epoch_log.csv` automatically.

This must be done on the main process only (already guaranteed since
`logger.log_epoch` guards on `is_main_process`). The timing wraps both the
train epoch and the val epoch together, giving total wall time per epoch
including data loading overhead.

### 2. No changes to `training.py`, `logging.py`, or `plot_overfit_summary.py`

`epoch_log.csv` already stores arbitrary `extra` keys; the plot script does
not use these columns so no changes needed there.

If a separate per-epoch time/memory plot is wanted later, add it to
`plot_overfit_summary.py` as a small figure — but that is out of scope here.

## Critical files
- `~/repositories/alphagenome-pytorch/scripts/finetune.py`
  — add `time.perf_counter()` bookends and `torch.cuda.max_memory_allocated`
    around the train + val calls inside the epoch loop

## Notes
- `torch.cuda.reset_peak_memory_stats` resets the high-water mark so each
  epoch is measured independently.
- Wall time includes val pass. If train-only time is needed later, a second
  timer can be added around the train call specifically.
- In multi-GPU (DDP) runs, `max_memory_allocated(device)` returns the peak
  for the current process's device; peak across all ranks is not aggregated
  (this is fine — all ranks typically use similar memory).

## Verification
1. Run 2 epochs of any overfitting config
2. Check `epoch_log.csv` contains `epoch_wall_time_s` and `peak_gpu_mem_gb`
3. Confirm values are reasonable (e.g., ~minutes per epoch, <80 GB)
