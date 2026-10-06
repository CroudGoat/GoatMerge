"""Command-line interface for GoatMerge (streaming task-arithmetic merge).

Subcommands:
  extract   T = W_source - W_base  -> task-vector directory
  merge     base + weighted TVs (or source models) -> merged model directory
  inspect   show metadata / manifest / fingerprint of a TV or merged model
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .consensus import ConsensusMethod
from .extract import extract_task_vector
from .hf import resolve_model_dir
from .inspect import inspect_dir, verify_against_base
from .merge import ModelEntry, MergeSettings, merge_model
from .metadata import build_merged_metadata, write_metadata
from .sparsify import SparsificationMethod


def _parse_entry(spec: str) -> ModelEntry:
    """Parse ``DIR:WEIGHT`` (last colon separates the weight)."""
    dir_str, _, weight_str = spec.rpartition(":")
    if not dir_str or not weight_str:
        raise argparse.ArgumentTypeError(f"bad entry {spec!r} (expected DIR:WEIGHT)")
    return ModelEntry(dir=Path(dir_str), kind="tv", weight=float(weight_str))


def _build_settings(args) -> MergeSettings:
    method = None
    if args.method:
        method = SparsificationMethod(args.method)
    return MergeSettings(
        density=args.density,
        method=method,
        n=args.n,
        m=args.m,
        gamma=args.gamma,
        epsilon=args.epsilon,
        rescale=not args.no_rescale,
        normalize=not args.no_normalize,
        lambda_=args.lambda_,
        consensus=ConsensusMethod(args.consensus),
        chunk_elements=args.chunk_elements,
    )


def cmd_extract(args) -> int:
    base = resolve_model_dir(args.base)
    source = resolve_model_dir(args.source)
    summary = extract_task_vector(base, source, Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_merge(args) -> int:
    base = resolve_model_dir(args.base)
    entries: list[ModelEntry] = []
    for spec in (args.tv or []):
        e = _parse_entry(spec)
        entries.append(
            ModelEntry(dir=resolve_model_dir(str(e.dir)), kind="tv", weight=e.weight)
        )
    for spec in (args.model or []):
        e = _parse_entry(spec)
        entries.append(
            ModelEntry(dir=resolve_model_dir(str(e.dir)), kind="model", weight=e.weight)
        )

    if not entries:
        print("error: no task vectors or models given", file=sys.stderr)
        return 2

    settings = _build_settings(args)

    # Fingerprint gate: every TV must have been extracted against this base.
    tv_dirs = [e.dir for e in entries if e.kind == "tv"]
    if tv_dirs and not args.skip_fingerprint_check:
        from .fingerprint import compute_base_fingerprint

        mismatches = verify_against_base(base, tv_dirs)
        if mismatches:
            print(
                "error: base fingerprint mismatch for: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 3

    summary = merge_model(base, entries, settings, Path(args.out))

    meta = build_merged_metadata(
        base_model=str(base),
        task_vectors=[
            {"dir": str(e.dir), "kind": e.kind, "weight": e.weight} for e in entries
        ],
        tensor_names=summary["tensor_names"],
        dtype=None,
        merge_settings={
            "density": settings.density,
            "method": settings.method.value if settings.method else None,
            "n": settings.n,
            "m": settings.m,
            "gamma": settings.gamma,
            "epsilon": settings.epsilon,
            "rescale": settings.rescale,
            "normalize": settings.normalize,
            "lambda": settings.lambda_,
            "consensus": settings.consensus.value,
            "chunk_elements": settings.chunk_elements,
        },
        skipped_tensors=summary["skipped_tensors"],
    )
    write_metadata(Path(args.out), meta)
    print(json.dumps({"out": str(Path(args.out)), **summary}, indent=2))
    return 0


def cmd_inspect(args) -> int:
    info = inspect_dir(Path(args.dir))
    print(json.dumps(info, indent=2))
    return 0


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(prog="goatmerge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="extract a task vector T = W_source - W_base")
    pe.add_argument("--base", required=True, help="base model dir or HF repo id")
    pe.add_argument("--source", required=True, help="source (fine-tuned) model dir or HF repo id")
    pe.add_argument("--out", required=True, help="output task-vector directory")
    pe.set_defaults(func=cmd_extract)

    pm = sub.add_parser("merge", help="merge base + weighted task vectors / models")
    pm.add_argument("--base", required=True, help="base model dir or HF repo id")
    pm.add_argument("--out", required=True, help="output merged-model directory")
    pm.add_argument("--tv", action="append", help="task-vector dir:weight (repeatable)")
    pm.add_argument("--model", action="append", help="source model dir:weight (repeatable)")
    pm.add_argument("--consensus", choices=["none", "sum", "count"], default="none")
    pm.add_argument("--density", type=float, default=1.0)
    pm.add_argument(
        "--method",
        choices=[m.value for m in SparsificationMethod],
        default=None,
    )
    pm.add_argument("--n", type=int, default=64, help="BS block top-n")
    pm.add_argument("--m", type=int, default=256, help="BS block size")
    pm.add_argument("--gamma", type=float, default=0.0)
    pm.add_argument("--epsilon", type=float, default=0.0)
    pm.add_argument("--no-rescale", action="store_true")
    pm.add_argument("--no-normalize", action="store_true")
    pm.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    pm.add_argument("--chunk-elements", type=int, default=None)
    pm.add_argument("--skip-fingerprint-check", action="store_true")
    pm.set_defaults(func=cmd_merge)

    pi = sub.add_parser("inspect", help="show metadata/manifest of a TV or merged model")
    pi.add_argument("--dir", required=True)
    pi.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
