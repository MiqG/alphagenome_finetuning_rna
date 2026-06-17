"""
Extract K562 splice-site usage from local AlphaGenome JAX TFRecords.

Reads all *.gz.tfrecord shards in --tfrecord-dir, extracts non-zero
K562 SSU values (+ and - strand), and writes a parquet table with
columns: chrom, pos, ssu_pos_strand, ssu_neg_strand.
"""
import argparse
import glob
import os

import certifi
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"]  = certifi.where()

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from alphagenome_research.io import bundles as bundles_lib
from alphagenome_research.io import dataset as dataset_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tfrecord-dir",      required=True)
    p.add_argument("--track-metadata",    required=True)
    p.add_argument("--output",            required=True)
    p.add_argument("--biosample-name",    default="K562")
    p.add_argument("--assay-title",       default="polyA plus RNA-seq")
    return p.parse_args()


def main():
    args = parse_args()

    track_meta = pd.read_parquet(args.track_metadata)
    ssu_tracks = (
        track_meta
        .query("output_type == 'splice_sites_usage'")
        .reset_index(drop=True)
    )

    sel = (
        ssu_tracks["biosample_name"].str.lower() == args.biosample_name.lower()
    ) & (
        ssu_tracks["assay_title"].str.lower() == args.assay_title.lower()
    )
    sample_tracks = ssu_tracks[sel]
    if len(sample_tracks) == 0:
        raise ValueError(
            "No SSU tracks found for biosample={} assay={}".format(
                args.biosample_name, args.assay_title
            )
        )

    idx_pos = sample_tracks[sample_tracks["track_strand"] == "+"].index[0]
    idx_neg = sample_tracks[sample_tracks["track_strand"] == "-"].index[0]
    print("+ strand column: {}, - strand column: {}".format(idx_pos, idx_neg))

    shards = sorted(glob.glob(os.path.join(args.tfrecord_dir, "*.gz.tfrecord")))
    if not shards:
        raise FileNotFoundError("No *.gz.tfrecord files in {}".format(args.tfrecord_dir))

    parser = dataset_lib._get_parse_function(bundles_lib.BundleName.SPLICE_SITES_USAGE)

    parts = []
    n_sites = 0
    with tqdm(shards, unit="shard", desc="Extracting K562 SSU") as pbar:
        for shard in pbar:
            ds = tf.data.TFRecordDataset(shard, compression_type="GZIP").map(parser)
            for element in ds:
                chrom = element["interval/chromosome"].numpy().decode()
                start = int(element["interval/start"].numpy())
                end   = int(element["interval/end"].numpy())

                # slice only the 2 K562 columns before materialising (367x less data)
                ssu = tf.gather(element["splice_site_usage"], [idx_pos, idx_neg], axis=1).numpy().astype(np.float32)

                # The SSU tensor covers the CENTRAL n_rows bases of the 1Mb input
                # window.  interval/start and interval/end span the full 1Mb input;
                # the output is cropped symmetrically, so the correct 0-based genomic
                # position for row i is: interval_start + crop_left + i.
                n_rows = ssu.shape[0]
                crop_left = (end - start - n_rows) // 2
                output_start = start + crop_left

                nonzero = (ssu[:, 0] > 0) | (ssu[:, 1] > 0)
                bp = np.where(nonzero)[0]

                if len(bp):
                    parts.append(pd.DataFrame({
                        "chrom":          chrom,
                        "pos":            output_start + bp,
                        "ssu_pos_strand": ssu[bp, 0],
                        "ssu_neg_strand": ssu[bp, 1],
                    }))
                    n_sites += len(bp)

            pbar.set_postfix(sites=n_sites)

    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["chrom", "pos", "ssu_pos_strand", "ssu_neg_strand"])
    print("Total non-zero K562 SSU sites: {}".format(len(df)))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output, index=False)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
