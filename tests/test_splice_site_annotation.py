"""Small, model-free regression cases for GTF site-label construction."""
import importlib.util
import ast
from pathlib import Path
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "src/scripts"
spec = importlib.util.spec_from_file_location("annotation", SCRIPTS / "splice_site_annotation.py")
annotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotation)


def frame(rows):
    return pd.DataFrame(rows, columns=[
        "Chromosome", "Start", "End", "Strand", "gene_id", "transcript_id"
    ]).assign(Feature="exon")


class AnnotationTests(unittest.TestCase):
    def test_both_collectors_share_helper_and_fingerprint_it(self):
        for path in (SCRIPTS / "collect_predictions.py", SCRIPTS.parents[1] /
                     "workflows/10-jax_finetuning/scripts/collect_predictions_jax.py"):
            source = path.read_text()
            tree = ast.parse(source)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "gtf_splice_sites"]
            self.assertEqual(len(calls), 1)
            self.assertEqual(ast.unparse(calls[0]),
                             "splice_site_annotation.gtf_splice_sites(gtf)")
            self.assertIn("splice_site_annotation.__file__", source)
            self.assertIn('"gtf_label_schema": splice_site_annotation.LABEL_SCHEMA', source)

    def test_strands_coordinates_and_terminal_exclusion(self):
        for strand in ("+", "-"):
            gtf = frame([("chr1", 100, 150, strand, "g", "t"),
                         ("chr1", 200, 250, strand, "g", "t")])
            self.assertEqual(annotation.gtf_splice_sites(gtf), {
                ("chr1", 150, strand, "donor"),
                ("chr1", 201, strand, "acceptor"),
            })

    def test_alternative_boundary_not_destroyed_by_gene_union(self):
        gtf = frame([("chr1", 100, 150, "+", "g", "t1"),
                     ("chr1", 200, 250, "+", "g", "t1"),
                     ("chr1", 100, 175, "+", "g", "t2"),
                     ("chr1", 200, 250, "+", "g", "t2")])
        self.assertEqual(annotation.gtf_splice_sites(gtf), {
            ("chr1", 150, "+", "donor"), ("chr1", 175, "+", "donor"),
            ("chr1", 201, "+", "acceptor"),
        })

    def test_never_pairs_single_exons_across_group_boundaries(self):
        gtf = frame([("chr1", 100, 150, "+", "g1", "t1"),
                     ("chr1", 200, 250, "+", "g1", "t2"),
                     ("chr2", 300, 350, "+", "g1", "t1"),
                     ("chr1", 400, 450, "-", "g1", "t1"),
                     ("chr1", 500, 550, "+", "g2", "t1")])
        self.assertEqual(annotation.gtf_splice_sites(gtf), set())

    def test_duplicate_exons_and_input_order(self):
        rows = [("chr1", 100, 150, "-", "g", "t"),
                ("chr1", 200, 250, "-", "g", "t")]
        self.assertEqual(annotation.gtf_splice_sites(frame(rows)),
                         annotation.gtf_splice_sites(frame((rows * 2)[::-1])))

    def test_no_intron_for_overlapping_or_touching_exons(self):
        for start in (149, 150):
            self.assertEqual(annotation.gtf_splice_sites(frame([
                ("chr1", 100, 150, "+", "g", "t"),
                ("chr1", start, 250, "+", "g", "t")
            ])), set())

    def test_unknown_strand_excluded(self):
        self.assertEqual(annotation.gtf_splice_sites(frame([
            ("chr1", 100, 150, ".", "g", "t"),
            ("chr1", 200, 250, ".", "g", "t")
        ])), set())

    def test_missing_or_invalid_metadata_rejected(self):
        gtf = frame([("chr1", 100, 150, "+", "g", "t")])
        for bad in (gtf.drop(columns="transcript_id"),
                    gtf.assign(transcript_id=None), gtf.assign(Start=-1),
                    gtf.assign(Start=100.5), gtf.assign(End=100)):
            with self.assertRaises(ValueError):
                annotation.gtf_splice_sites(bad)


if __name__ == "__main__":
    unittest.main()
