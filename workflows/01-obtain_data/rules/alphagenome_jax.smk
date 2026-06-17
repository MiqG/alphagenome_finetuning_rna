ALPHAGENOME_JAX_FOLD_SPLITS = ["FOLD_1"] # ["ALL_FOLDS", "FOLD_0", "FOLD_1", "FOLD_2", "FOLD_3"]
ALPHAGENOME_JAX_ORGANISMS   = ["HOMO_SAPIENS"]
ALPHAGENOME_JAX_SUBSETS     = ["VALID"] #["VALID", "TEST"] # "TRAIN", 
ALPHAGENOME_JAX_BUNDLES     = [
    # "ATAC", "CAGE", "CHIP_HISTONE", "CHIP_TF", "CONTACT_MAPS","DNASE", "PROCAP", 
    # "RNA_SEQ", "SPLICE_JUNCTIONS", "SPLICE_SITES", "SPLICE_SITE_POSITIONS", 
    "SPLICE_SITE_USAGE",
]


rule download_alphagenome_jax_bundle:
    """Download one AlphaGenome JAX TFRecord bundle directory from GCS."""
    output:
        done = os.path.join(
            config["alphagenome_jax"]["path"],
            "{fold_split}", "{organism}", "{subset}", "{bundle}", ".done",
        ),
    params:
        gcs_src   = lambda wc: "{gcs}/{fs}/{org}/{sub}/{bun}/".format(
            gcs=config["alphagenome_jax"]["gcs_path"],
            fs=wc.fold_split,
            org=wc.organism,
            sub=wc.subset,
            bun=wc.bundle,
        ),
        local_dir = lambda wc: os.path.join(
            config["alphagenome_jax"]["path"],
            wc.fold_split, wc.organism, wc.subset, wc.bundle,
        ),
    threads: 4
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 6 * 60,
        memory    = 8,
    shell:
        """
        set -eo pipefail

        mkdir -p {params.local_dir}

        gsutil -m cp "{params.gcs_src}*.gz.tfrecord" {params.local_dir}/

        touch {output.done}
        echo "Done!"
        """


rule extract_k562_ssu_from_tfrecords:
    """Extract K562 SSU values from local SPLICE_SITE_USAGE TFRecords into a parquet table."""
    input:
        done = os.path.join(
            config["alphagenome_jax"]["path"],
            "{fold_split}", "{organism}", "{subset}", "SPLICE_SITE_USAGE", ".done",
        ),
        track_metadata = config["examples"]["alphagenome"]["track_metadata"],
    output:
        parquet = os.path.join(
            config["alphagenome_jax"]["path"],
            "{fold_split}", "{organism}", "{subset}", "k562_ssu.parquet",
        ),
    params:
        tfrecord_dir = lambda wc: os.path.join(
            config["alphagenome_jax"]["path"],
            wc.fold_split, wc.organism, wc.subset, "SPLICE_SITE_USAGE",
        ),
        script = os.path.join(SCRIPTS_DIR, "extract_k562_ssu_tfrecords.py"),
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 4 * 60,
        memory    = 32,
    conda:
        "alphagenome"
    shell:
        """
        export LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:${{LD_LIBRARY_PATH:-}}"
        python {params.script} \
            --tfrecord-dir {params.tfrecord_dir} \
            --track-metadata {input.track_metadata} \
            --output {output.parquet}
        """

rule merge_k562_ssu:
    """Merge per-split K562 SSU parquets into one table with provenance columns."""
    input:
        parquets = expand(
            os.path.join(
                config["alphagenome_jax"]["path"],
                "{fold_split}", "{organism}", "{subset}", "k562_ssu.parquet",
            ),
            fold_split=ALPHAGENOME_JAX_FOLD_SPLITS,
            organism=ALPHAGENOME_JAX_ORGANISMS,
            subset=ALPHAGENOME_JAX_SUBSETS,
        ),
    output:
        merged = os.path.join(config["alphagenome_jax"]["path"], "k562_ssu_merged.parquet"),
    resources:
        gres      = "none",
        partition = "genoa64",
        runtime   = 30,
        memory    = 16,
    run:
        import pandas as pd
        base = config["alphagenome_jax"]["path"]
        parts = []
        for fold_split in ALPHAGENOME_JAX_FOLD_SPLITS:
            for organism in ALPHAGENOME_JAX_ORGANISMS:
                for subset in ALPHAGENOME_JAX_SUBSETS:
                    path = os.path.join(base, fold_split, organism, subset, "k562_ssu.parquet")
                    df = pd.read_parquet(path)
                    df["fold_split"] = fold_split
                    df["organism"]   = organism
                    df["subset"]     = subset
                    parts.append(df)
        merged = pd.concat(parts, ignore_index=True)
        merged.to_parquet(output.merged, index=False)
        print("Merged {} rows from {} files.".format(len(merged), len(parts)))
