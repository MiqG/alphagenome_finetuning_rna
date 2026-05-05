# Plan: track auPRC for splice_site head per epoch

## Context
README step 3 (`original` and `debug_splice_sites` groups) requires storing
"auPRC against ground-truth categorical labels" for the splice_site modality.
`validate_multihead` currently computes only Pearson R for all heads; the
classification head's per-class average precision is never stored.

The splice_site classification head predicts 5 classes:
no_site / donor_pos / acceptor_pos / donor_neg / acceptor_neg.
`targets_dict["probs"]` holds one-hot ground-truth (B, S, 5) and
`predictions[1]` holds raw logits (B, S, 5) — both are already available
inside `_compute_splice_loss` and `_extract_splice_pearson_pairs`.

## Changes

### 1. `training.py` — compute auPRC for classification head in `validate_multihead`

After the existing `accumulated_splice` Pearson R block (around line 2111),
add an auPRC accumulation block for `SpliceSitesClassificationHead`:

```python
# Accumulate (logits, true_onehot) for auPRC across batches
accumulated_cls[modality]["logits"].append(predictions_scaled[1].float().cpu())
accumulated_cls[modality]["true"].append(targets_dict["probs"].float().cpu())
```

Then after the loop, compute per-class AP and macro average:

```python
from sklearn.metrics import average_precision_score
import torch.nn.functional as F

for modality in heads:
    head_module = heads[modality].module if hasattr(...) else heads[modality]
    if not isinstance(head_module, SpliceSitesClassificationHead):
        continue
    if not accumulated_cls[modality]["logits"]:
        continue
    all_logits = torch.cat(accumulated_cls[modality]["logits"], dim=0)  # (N, S, 5)
    all_true   = torch.cat(accumulated_cls[modality]["true"],   dim=0)  # (N, S, 5)
    probs = F.softmax(all_logits, dim=-1)
    # Flatten to (N*S, 5); keep only positions where any class is active
    mask = all_true.any(dim=-1).reshape(-1)
    probs_flat = probs.reshape(-1, 5)[mask].numpy()
    true_flat  = all_true.reshape(-1, 5)[mask].numpy()
    CLASS_NAMES = ["no_site", "donor_pos", "acceptor_pos", "donor_neg", "acceptor_neg"]
    for i, cls_name in enumerate(CLASS_NAMES):
        ap = average_precision_score(true_flat[:, i], probs_flat[:, i])
        metrics[f"{modality}_auprc_{cls_name}"] = ap
    # Macro average over splice-site classes only (exclude no_site = class 0)
    macro_ap = average_precision_score(true_flat[:, 1:], probs_flat[:, 1:], average="macro")
    metrics[f"{modality}_auprc_macro"] = macro_ap
```

`sklearn` is already available in the conda env (used by other metrics scripts).
Initialize `accumulated_cls` as `defaultdict(lambda: {"logits": [], "true": []})`.

### 2. `finetune.py` — no changes needed

The `extra` dict is built from all `val_metrics` keys that don't end in `_values`
and don't contain "pearson" — auPRC keys match neither filter, so they
already flow into `epoch_log.csv` automatically.

Wait: actually the current filter is:
```python
elif "pearson" in key:
    extra[key] = val
else:
    extra[f"val_loss_{key}"] = val
```

auPRC keys will be prefixed with `val_loss_` which is confusing.
Fix: add an extra condition before the else:

```python
elif "auprc" in key:
    extra[key] = val   # store as-is, no val_loss_ prefix
```

### 3. `plot_overfit_summary.py` — add auPRC page

Add a Page 5 (or new figure) showing per-class auPRC across runs as a bar/strip:

```python
AUPRC_COLS = {
    "splice_site_auprc_donor_pos":     "donor+",
    "splice_site_auprc_acceptor_pos":  "acceptor+",
    "splice_site_auprc_donor_neg":     "donor-",
    "splice_site_auprc_acceptor_neg":  "acceptor-",
    "splice_site_auprc_macro":         "macro",
}

def page_auprc(epoch_dfs, runs, colors):
    # Line plot of macro auPRC over epochs (like page_val_correlations)
    # Only render if splice_site_auprc_macro column exists
    ...
```

Add this page inside the `with PdfPages` block in `main()`, skipping if
the column is absent (so older runs without the metric don't break).

## Critical files
- `~/repositories/alphagenome-pytorch/src/alphagenome_pytorch/extensions/finetuning/training.py`
  — add auPRC accumulation + computation in `validate_multihead`
- `~/repositories/alphagenome-pytorch/scripts/finetune.py`
  — fix `extra` key routing for auPRC (avoid `val_loss_` prefix)
- `src/scripts/plot_overfit_summary.py`
  — add auPRC page; guard with column-existence check

## Verification
1. Run one epoch of a `debug_splice_sites` config with `--modality-weights splice_site:1.0,...`
2. Check `epoch_log.csv` contains `splice_site_auprc_macro`, `splice_site_auprc_donor_pos`, etc.
3. Run `plot_overfit_summary.py --run-dirs <run_dir> --output /tmp/test.pdf` and confirm Page 5 renders
