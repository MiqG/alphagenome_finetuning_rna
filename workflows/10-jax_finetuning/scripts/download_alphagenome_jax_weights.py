"""One-time cache of the Kaggle-hosted AlphaGenome JAX checkpoint.

``alphagenome_ft.create_model_with_heads`` (via
``alphagenome_research.model.dna_model.create_from_kaggle``) downloads
pretrained weights through ``kagglehub``, which needs Kaggle credentials and
internet access. SLURM GPU compute nodes on this cluster typically have
neither, so this script runs once (on the login node, where credentials and
internet are available) to populate the kagglehub cache and record the
resulting local checkpoint directory. Downstream training jobs then pass that
local path via ``--checkpoint-path`` to ``create_model_with_heads``, so the
GPU job itself never talks to Kaggle.

Requires Kaggle credentials to already be configured (``KAGGLE_USERNAME``/
``KAGGLE_KEY`` env vars, or ``~/.kaggle/kaggle.json``) — see
https://github.com/Kaggle/kagglehub#authenticate.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-version", default="fold_1",
        help=(
            "AlphaGenome JAX model version to cache (default: fold_1, matching "
            "alphagenome_pytorch's model_fold_1.safetensors — NOT all_folds, "
            "which would leak our FOLD_1 test intervals into the pretrained "
            "backbone's own training data)."
        ),
    )
    parser.add_argument(
        "--output-path-file", required=True, type=Path,
        help="Text file to write the resolved local checkpoint directory to.",
    )
    args = parser.parse_args()

    import kagglehub

    checkpoint_path = kagglehub.model_download(
        f"google/alphagenome/jax/{args.model_version.lower()}"
    )
    print(f"Cached AlphaGenome JAX checkpoint ({args.model_version}) at: {checkpoint_path}")

    args.output_path_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_path_file.write_text(str(checkpoint_path) + "\n")


if __name__ == "__main__":
    main()
