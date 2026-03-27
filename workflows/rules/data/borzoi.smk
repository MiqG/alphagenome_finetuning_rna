rule borzoi_download_support:
    """Download Borzoi pretrained trunks and hg38 support files from GCS."""
    output:
        blacklist = config["borzoi"]["support"]["blacklist"],
        sequences_bed = config["borzoi"]["support"]["sequences_bed"],
        trunks = config["borzoi"]["pretrained_trunks"],
    params:
        hg38_dir = os.path.dirname(config["borzoi"]["support"]["blacklist"]),
        trainsplit_dir = os.path.dirname(config["borzoi"]["support"]["sequences_bed"]),
        trunks_dir = os.path.dirname(config["borzoi"]["pretrained_trunks"][0]),
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 2*60,
        memory = 4
    conda:
        "borzoi"
    shell:
        """
        set -eo pipefail

        gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/hg38/ {params.hg38_dir}
        gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/trainsplit/ {params.trainsplit_dir}
        gsutil cp -r gs://scbasset_tutorial_data/baskerville_transfer/pretrain_trunks/ {params.trunks_dir}

        echo "Done!"
        """
