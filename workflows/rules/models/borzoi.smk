BORZOI_CFG = config["borzoi"]
BORZOI_DATA_DIR = config["finetuning"]["borzoi"]["sf3b1mut"]["data_dir"]
BORZOI_OUT_DIR = config["finetuning"]["borzoi"]["sf3b1mut"]["output_dir"]
BORZOI_N_REPS = BORZOI_CFG["num_reps"]
BORZOI_REPS = [str(i) for i in range(BORZOI_N_REPS)]

BASKERVILLE_PATH = BORZOI_CFG["baskerville_path"]
BW_W5_SCRIPT = os.path.join(BASKERVILLE_PATH, "src/baskerville/scripts/utils/bw_w5.py")
SETUP_FOLDS_SCRIPT = os.path.join(BASKERVILLE_PATH, "docs/transfer_human/setup_folds.py")


rule borzoi_bw_to_w5:
    """Convert a strand-specific bigwig file to compressed w5 (HDF5) format."""
    input:
        bw_done = os.path.join(DATA_DIR, "STAR", ".done_make_bw", "{sample}-{strand}"),
    output:
        w5 = os.path.join(BORZOI_DATA_DIR, "w5", "{sample}.{strand}.w5"),
    params:
        bw_file = lambda wildcards: os.path.join(
            DATA_DIR, "STAR", wildcards.sample,
            "second_pass.Aligned.sortedByCoord.out.filtered.{strand}.bw".format(
                strand=wildcards.strand
            ),
        ),
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 2*60,
        memory = 8
    conda:
        "../envs/borzoi.yaml"
    shell:
        """
        python {BW_W5_SCRIPT} {params.bw_file} {output.w5}
        echo "Done!"
        """


rule borzoi_make_targets:
    """Build targets.txt and params.json for Borzoi transfer learning."""
    input:
        w5_files = [
            os.path.join(BORZOI_DATA_DIR, "w5", "{sample}.{strand}.w5").format(
                sample=s, strand=st
            )
            for s in SAMPLES
            for st in STRANDS
        ],
    output:
        targets = os.path.join(BORZOI_DATA_DIR, "targets.txt"),
        params_json = config["finetuning"]["borzoi"]["sf3b1mut"]["params_json"],
    params:
        clip = config["finetuning"]["borzoi"]["sf3b1mut"]["clip"],
        clip_soft = config["finetuning"]["borzoi"]["sf3b1mut"]["clip_soft"],
        scale = config["finetuning"]["borzoi"]["sf3b1mut"]["scale"],
        template_json = os.path.join(
            BASKERVILLE_PATH, "tests/data/transfer/json/borzoi_lora.json"
        ),
    run:
        import json
        import pandas as pd

        # --- targets.txt ---
        records = []
        for sample in SAMPLES:
            for strand in STRANDS:
                strand_symbol = "+" if strand == "forward" else "-"
                records.append({
                    "identifier": f"{sample}{strand_symbol}",
                    "file": os.path.join(BORZOI_DATA_DIR, "w5", f"{sample}.{strand}.w5"),
                    "clip": params.clip,
                    "clip_soft": params.clip_soft,
                    "scale": params.scale,
                    "sum_stat": "sum_sqrt",
                    "description": f"{sample} RNA-seq {strand} strand",
                })

        # assign strand_pair: index of the complementary strand track
        for i, rec in enumerate(records):
            base = rec["identifier"][:-1]
            paired_symbol = "-" if rec["identifier"].endswith("+") else "+"
            for j, other in enumerate(records):
                if other["identifier"] == f"{base}{paired_symbol}":
                    rec["strand_pair"] = j
                    break

        df = pd.DataFrame(records)[
            ["identifier", "file", "clip", "clip_soft", "scale",
             "sum_stat", "strand_pair", "description"]
        ]
        df.to_csv(output.targets, sep="\t")

        # --- params.json: update head units to match number of tracks ---
        with open(params.template_json) as f:
            params_json = json.load(f)

        params_json["model"]["head_human"]["units"] = len(records)

        os.makedirs(os.path.dirname(output.params_json), exist_ok=True)
        with open(output.params_json, "w") as f:
            json.dump(params_json, f, indent=4)

        print("Done!")


rule borzoi_make_tfrecords:
    """Create TFRecords from w5 files using hound_data.py."""
    input:
        targets = os.path.join(BORZOI_DATA_DIR, "targets.txt"),
        genome = config["gencode"]["paths"]["fasta"],
        blacklist = BORZOI_CFG["support"]["blacklist"],
        sequences_bed = BORZOI_CFG["support"]["sequences_bed"],
    output:
        done = touch(os.path.join(BORZOI_DATA_DIR, "tfr", ".done")),
    params:
        out_dir = os.path.join(BORZOI_DATA_DIR, "tfr"),
        folds = config["finetuning"]["borzoi"]["sf3b1mut"]["folds"],
    threads: 32
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 12*60,
        memory = 64
    conda:
        "../envs/borzoi.yaml"
    shell:
        """
        set -eo pipefail

        mkdir -p {params.out_dir}
        cp {input.sequences_bed} {params.out_dir}/

        hound_data.py \
            --restart \
            -c 163840 \
            -d 2 \
            -f {params.folds} \
            -l 524288 \
            -p {threads} \
            -r 256 \
            -w 32 \
            --local \
            -b {input.blacklist} \
            -o {params.out_dir} \
            {input.genome} \
            {input.targets}

        echo "Done!"
        """


rule borzoi_setup_folds:
    """Set up cross-fold directory structure for transfer learning."""
    input:
        tfr_done = os.path.join(BORZOI_DATA_DIR, "tfr", ".done"),
        params_json = config["finetuning"]["borzoi"]["sf3b1mut"]["params_json"],
    output:
        done = touch(os.path.join(BORZOI_OUT_DIR, ".done_setup")),
    params:
        tfr_dir = os.path.join(BORZOI_DATA_DIR, "tfr"),
        out_dir = BORZOI_OUT_DIR,
        folds = config["finetuning"]["borzoi"]["sf3b1mut"]["folds"],
    threads: 1
    resources:
        gres = "none",
        partition = "gpu_diasfrazer",
        runtime = 1*60,
        memory = 4
    conda:
        "../envs/borzoi.yaml"
    shell:
        """
        set -eo pipefail

        python {SETUP_FOLDS_SCRIPT} \
            -o {params.out_dir} \
            -f {params.folds} \
            {input.params_json} \
            {params.tfr_dir}

        echo "Done!"
        """


rule borzoi_transfer:
    """Run Borzoi LoRA transfer learning for one replicate trunk."""
    input:
        setup_done = os.path.join(BORZOI_OUT_DIR, ".done_setup"),
        trunk = lambda wildcards: BORZOI_CFG["pretrained_trunks"][int(wildcards.rep)],
        params_json = config["finetuning"]["borzoi"]["sf3b1mut"]["params_json"],
    output:
        done = touch(os.path.join(BORZOI_OUT_DIR, "rep{rep}", ".done")),
    params:
        out_dir = os.path.join(BORZOI_OUT_DIR, "rep{rep}"),
        data_dir = os.path.join(BORZOI_OUT_DIR, "f3c0", "data0"),
    threads: 1
    resources:
        gres = "gpu:1",
        partition = "gpu_diasfrazer",
        runtime = 48*60,
        memory = 40
    conda:
        "../envs/borzoi.yaml"
    shell:
        """
        set -eo pipefail

        hound_transfer.py \
            -o {params.out_dir} \
            --trunk \
            --restore {input.trunk} \
            {input.params_json} \
            {params.data_dir}

        echo "Done!"
        """
