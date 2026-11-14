import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .analyzer import analyze_spectrum
from .evaluation import evaluate_retrieval
from .validation import validate_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paravec", description="Analyze vector spectra and bit allocations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze a NumPy matrix of embeddings")
    analyze.add_argument("input", type=Path, help="input .npy matrix with shape (vectors, dimensions)")
    analyze.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128])
    analyze.add_argument("--output", type=Path, help="write the JSON report to this path")
    validate = subparsers.add_parser("validate", help="validate spectral bounds and staged search on a NumPy matrix")
    validate.add_argument("input", type=Path, help="input .npy matrix with shape (vectors, dimensions)")
    validate.add_argument("--queries", type=int, default=32)
    validate.add_argument("--k", type=int, default=10)
    validate.add_argument("--bit-budget", type=int, default=32)
    validate.add_argument("--seed", type=int, default=42)
    validate.add_argument("--output", type=Path, help="write the JSON report to this path")
    evaluate = subparsers.add_parser("evaluate", help="measure held-out recall and storage estimates for packed search")
    evaluate.add_argument("input", type=Path, help="input .npy matrix with shape (vectors, dimensions)")
    evaluate.add_argument("--budgets", type=int, nargs="+", default=[32, 64, 128])
    evaluate.add_argument("--queries", type=int, default=32)
    evaluate.add_argument("--k", type=int, default=10)
    evaluate.add_argument("--train-fraction", type=float, default=0.8)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--output", type=Path, help="write the JSON report to this path")
    return parser


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        vectors = np.load(args.input, allow_pickle=False)
        if args.command == "analyze":
            report = analyze_spectrum(vectors, bit_budgets=args.budgets)
        elif args.command == "validate":
            report = validate_dataset(
                vectors,
                query_count=args.queries,
                k=args.k,
                bit_budget=args.bit_budget,
                seed=args.seed,
            )
        else:
            report = evaluate_retrieval(
                vectors,
                bit_budgets=args.budgets,
                query_count=args.queries,
                k=args.k,
                train_fraction=args.train_fraction,
                seed=args.seed,
            )
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        if args.command == "validate" and report["status"] != "passed":
            return 1
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    return 0