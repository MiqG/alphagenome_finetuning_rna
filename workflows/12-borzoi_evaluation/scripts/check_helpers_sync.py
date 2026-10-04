#!/usr/bin/env python
"""Fail if rna_eval_helpers.py drifts from src/scripts/collect_predictions.py.

Compares the AST of every duplicated helper with the original. Run in the
alphagenome_pytorch env (any env with python is enough: both files are only parsed):

    python workflows/12-borzoi_evaluation/scripts/check_helpers_sync.py
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = os.path.normpath(os.path.join(HERE, "..", "..", "..", "src", "scripts", "collect_predictions.py"))
COPY = os.path.join(HERE, "rna_eval_helpers.py")


def definitions(path):
    with open(path) as handle:
        tree = ast.parse(handle.read())
    return {
        node.name: ast.dump(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


original, copy = definitions(ORIGINAL), definitions(COPY)
problems = [
    name for name in copy
    if name not in original or original[name] != copy[name]
]
if problems:
    sys.exit("Out of sync with collect_predictions.py: {}".format(", ".join(problems)))
print("OK: {} helpers identical to collect_predictions.py".format(len(copy)))
