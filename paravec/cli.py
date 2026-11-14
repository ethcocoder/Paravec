import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .analyzer import analyze_spectrum


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paravec", description="Analyze vector spectra and bit allocations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a NumPy matrix of embeddings")
    analyze.add_argument("input", type=Path, help="input .npy matrix with shape (vectors, dimensions)")
    analyze.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    analyze.add_argument("--output", type=Path, help="write the JSON report to this path")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        vectors = np.load(args.input, allow_pickle=False)
        report = analyze_spectrum(vectors, bit_budgets=args.budgets)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0