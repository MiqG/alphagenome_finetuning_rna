"""Model-free checks that cached rows cannot mix old and corrected GTF labels."""
import ast
import importlib.util
import os
from pathlib import Path
import pickle
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "src/scripts"
spec = importlib.util.spec_from_file_location("provenance", SCRIPTS / "evaluation_provenance.py")
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)
COLLECTORS = (SCRIPTS / "collect_predictions.py",
              ROOT / "workflows/10-jax_finetuning/scripts/collect_predictions_jax.py")


def progress_functions(path):
    # Execute only pure progress helpers; no model/framework imports or init.
    tree = ast.parse(path.read_text())
    names = {"_progress_dir", "_load_progress", "_save_progress"}
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in funcs} == names
    namespace = {"os": os, "pickle": pickle}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class LabelResumeTests(unittest.TestCase):
    def test_collectors_roundtrip_and_reject_mismatched_state(self):
        for path in COLLECTORS:
            with self.subTest(collector=path.name), tempfile.TemporaryDirectory() as output:
                funcs = progress_functions(path)
                funcs["_save_progress"](output, {1, 3}, {"sites": [{"in_gtf": True}]}, {}, "new-labels")
                done, state = funcs["_load_progress"](output, "new-labels")
                self.assertEqual(done, {1, 3})
                self.assertEqual(state["row_lists"], {"sites": [{"in_gtf": True}]})
                with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                    funcs["_load_progress"](output, "old-labels")

    def test_collectors_reject_legacy_state_without_fingerprint(self):
        for path in COLLECTORS:
            with self.subTest(collector=path.name), tempfile.TemporaryDirectory() as output:
                funcs = progress_functions(path)
                state = Path(funcs["_progress_dir"](output)) / "state.pkl"
                state.write_bytes(pickle.dumps({"done_intervals": {0}}))
                with self.assertRaises(RuntimeError):
                    funcs["_load_progress"](output, "new-labels")

    def test_guard_refuses_legacy_progress_before_reading_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / ".progress").mkdir()
            (output / ".progress/state.pkl").write_bytes(b"not a pickle; must not load")
            with self.assertRaisesRegex(RuntimeError, "Unsafe legacy resume"):
                provenance.ensure_provenance(str(output), {"fingerprint_sha256": "new"})

    def test_changed_schema_gtf_and_helper_code_refuse_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gtf, helper, output = root / "gtf", root / "helper.py", root / "output"
            gtf.write_text("original annotation")
            helper.write_text("original helper")
            def fingerprint(schema="new"):
                return provenance.build_fingerprint(framework="test", settings={"gtf_label_schema": schema},
                                                     paths={"gtf": gtf}, code_paths=[helper])
            original = fingerprint()
            provenance.ensure_provenance(str(output), original)
            self.assertEqual(provenance.ensure_provenance(str(output), fingerprint()), original["fingerprint_sha256"])
            with self.assertRaises(RuntimeError):
                provenance.ensure_provenance(str(output), fingerprint("old"))
            gtf.write_text("changed annotation")
            with self.assertRaises(RuntimeError):
                provenance.ensure_provenance(str(output), fingerprint())
            # Establish a fresh baseline so the code-only mutation is isolated
            # from the preceding GTF contents/mtime change.
            code_output = root / "output-code"
            provenance.ensure_provenance(str(code_output), fingerprint())
            helper.write_text("changed helper")
            with self.assertRaises(RuntimeError):
                provenance.ensure_provenance(str(code_output), fingerprint())

    def test_collectors_guard_before_model_loading_and_pass_fingerprint(self):
        for path in COLLECTORS:
            with self.subTest(collector=path.name):
                tree = ast.parse(path.read_text())
                main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
                calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)]
                guards = [n for n in calls if ast.unparse(n.func) == "evaluation_provenance.ensure_provenance"]
                models = [n for n in calls if ast.unparse(n.func).startswith("load_finetuned_model")]
                self.assertEqual(len(guards), 1)
                self.assertEqual(len(models), 1)
                self.assertLess(guards[0].lineno, models[0].lineno)
                progress = [n for n in calls if ast.unparse(n.func) in ("_load_progress", "_save_progress")]
                self.assertEqual(len(progress), 3)
                for call in progress:
                    self.assertEqual(ast.unparse(call.args[-1]), "fingerprint_sha256")
                self.assertIn('"transcript_id"', path.read_text())

    def test_preserves_new_packaged_jax_import(self):
        tree = ast.parse(COLLECTORS[1].read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and n.module == "alphagenome_ft.finetune.combined_data"]
        self.assertTrue(any(a.name == "compute_track_means" for n in imports for a in n.names))


if __name__ == "__main__":
    unittest.main()
