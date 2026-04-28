import os

HEPG2_DATA_DIR = config["rnaseq"]["hepg2"]["path"]

ENDS_HEPG2 = ["1", "2"]

# Paired-end FASTQ file accessions per sample (experiment_rep -> {end: ENCFF_accession})
SAMPLES_HEPG2 = {
    "ENCSR000CPE_rep1": {"1": "ENCFF065PJM", "2": "ENCFF217MPH"},
    "ENCSR000CPE_rep2": {"1": "ENCFF098REH", "2": "ENCFF830NYV"},
    "ENCSR181ZGR_rep1": {"1": "ENCFF581ZGH", "2": "ENCFF177BYA"},
    "ENCSR181ZGR_rep2": {"1": "ENCFF143QGY", "2": "ENCFF861HZL"},
    "ENCSR000EYR_rep1": {"1": "ENCFF995HXY", "2": "ENCFF482WGP"},
}

# Flatten to {ENCFF_id: download_url}
URLS_HEPG2 = {
    accession: "https://www.encodeproject.org/files/{acc}/@@download/{acc}.fastq.gz".format(acc=accession)
    for sample in SAMPLES_HEPG2.values()
    for accession in sample.values()
}


rule download_hepg2_fastq:
    params:
        sample = "{sample}",
        end = "{end}",
        url = lambda wildcards: URLS_HEPG2[SAMPLES_HEPG2[wildcards.sample][wildcards.end]],
        fastqs_dir = os.path.join(HEPG2_DATA_DIR, "fastqs"),
    output:
        download_done = os.path.join(HEPG2_DATA_DIR, "fastqs", ".done", "{sample}_{end}"),
    threads: 1
    resources:
        gres = "none",
        partition = "genoa64",
        runtime = 60 * 24,  # minutes
        memory = 2,
    conda:
        "alphagenome_pytorch"
    shell:
        """
        set -eo pipefail

        echo "Downloading {params.sample} end {params.end}..."

        wget --user-agent="Chrome" \
             --no-check-certificate \
             {params.url} \
             -O {params.fastqs_dir}/{params.sample}_{params.end}.fastq.gz

        touch {output.download_done}
        echo "Finished downloading {params.sample} end {params.end}."
        echo $(date)

        echo "Done!"
        """
