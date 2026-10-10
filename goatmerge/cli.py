"""Command-line interface for GoatMerge (streaming task-arithmetic merge).

Subcommands:
  extract   T = W_source - W_base  -> task-vector directory
  merge     base + weighted TVs (or source models) -> merged model directory
  inspect   show metadata / manifest / fingerprint of a TV or merged model

YAML recipe files:
  The ``merge`` subcommand accepts ``--config / -c`` pointing to a YAML file
  that supplies base, out, tv/model entries, and all tuning parameters.
  Explicit CLI flags override YAML values.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from .consensus import ConsensusMethod
from .extract import extract_task_vector
from .hf import resolve_model_dir
from .inspect import inspect_dir, verify_against_base
from .merge import ModelEntry, MergeSettings, merge_model
from .merge_method import MergeMethod
from .metadata import build_merged_metadata, write_metadata
from .sparsify import SparsificationMethod


def _parse_entry(spec: str) -> ModelEntry:
    """Parse ``DIR:WEIGHT`` (last colon separates the weight)."""
    dir_str, _, weight_str = spec.rpartition(":")
    if not dir_str or not weight_str:
        raise argparse.ArgumentTypeError(f"bad entry {spec!r} (expected DIR:WEIGHT)")
    return ModelEntry(dir=Path(dir_str), kind="tv", weight=float(weight_str))


def _load_yaml_config(path: str) -> dict:
    """Load a YAML recipe file and return its contents as a dict."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_param(cli_val, yaml_dict: dict, key: str, default):
    """Return cli_val if not None (explicitly set), else yaml[key] if present, else default."""
    if cli_val is not None:
        return cli_val
    if key in yaml_dict:
        return yaml_dict[key]
    return default


def _build_settings(args, yaml_dict: dict) -> MergeSettings:
    merge_method = _resolve_param(args.merge_method, yaml_dict, "merge_method", "gta")
    density = _resolve_param(args.density, yaml_dict, "density", 1.0)
    method_str = _resolve_param(args.method, yaml_dict, "method", None)
    n = _resolve_param(args.n, yaml_dict, "n", 64)
    m = _resolve_param(args.m, yaml_dict, "m", 256)
    gamma = _resolve_param(args.gamma, yaml_dict, "gamma", 0.0)
    epsilon = _resolve_param(args.epsilon, yaml_dict, "epsilon", 0.0)
    no_rescale = _resolve_param(args.no_rescale, yaml_dict, "no_rescale", False)
    no_normalize = _resolve_param(args.no_normalize, yaml_dict, "no_normalize", False)
    lambda_ = _resolve_param(args.lambda_, yaml_dict, "lambda", 1.0)
    consensus = _resolve_param(args.consensus, yaml_dict, "consensus", "none")
    chunk_elements = _resolve_param(args.chunk_elements, yaml_dict, "chunk_elements", None)

    method = None
    if method_str is not None:
        method = SparsificationMethod(method_str)

    return MergeSettings(
        merge_method=MergeMethod(merge_method),
        density=density,
        method=method,
        n=n,
        m=m,
        gamma=gamma,
        epsilon=epsilon,
        rescale=not no_rescale,
        normalize=not no_normalize,
        lambda_=lambda_,
        consensus=ConsensusMethod(consensus),
        chunk_elements=chunk_elements,
    )


def cmd_extract(args) -> int:
    base = resolve_model_dir(args.base)
    source = resolve_model_dir(args.source)
    summary = extract_task_vector(base, source, Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_merge(args) -> int:
    # Load YAML config if provided
    yaml_dict = {}
    if args.config:
        yaml_dict = _load_yaml_config(args.config)

    # Resolve base and out (CLI > YAML > error)
    base_str = args.base if args.base is not None else yaml_dict.get("base")
    out_str = args.out if args.out is not None else yaml_dict.get("out")
    if not base_str:
        print("error: --base (or 'base' in config) is required", file=sys.stderr)
        return 2
    if not out_str:
        print("error: --out (or 'out' in config) is required", file=sys.stderr)
        return 2

    base = resolve_model_dir(base_str)

    # Collect entries: CLI flags take priority, then YAML
    tv_specs: list[str] = []
    model_specs: list[str] = []

    if args.tv:
        tv_specs.extend(args.tv)
    elif "tv" in yaml_dict:
        for entry in yaml_dict["tv"]:
            if isinstance(entry, str):
                tv_specs.append(entry)
            elif isinstance(entry, dict) and "dir" in entry:
                tv_specs.append(f"{entry['dir']}:{entry.get('weight', 1.0)}")

    if args.model:
        model_specs.extend(args.model)
    elif "model" in yaml_dict:
        for entry in yaml_dict["model"]:
            if isinstance(entry, str):
                model_specs.append(entry)
            elif isinstance(entry, dict) and "dir" in entry:
                model_specs.append(f"{entry['dir']}:{entry.get('weight', 1.0)}")

    entries: list[ModelEntry] = []
    for spec in tv_specs:
        e = _parse_entry(spec)
        entries.append(
            ModelEntry(dir=resolve_model_dir(str(e.dir)), kind="tv", weight=e.weight)
        )
    for spec in model_specs:
        e = _parse_entry(spec)
        entries.append(
            ModelEntry(dir=resolve_model_dir(str(e.dir)), kind="model", weight=e.weight)
        )

    if not entries:
        print("error: no task vectors or models given", file=sys.stderr)
        return 2

    settings = _build_settings(args, yaml_dict)

    # Fingerprint gate
    skip_fp = _resolve_param(args.skip_fingerprint_check, yaml_dict, "skip_fingerprint_check", False)
    tv_dirs = [e.dir for e in entries if e.kind == "tv"]
    if tv_dirs and not skip_fp:
        from .fingerprint import compute_base_fingerprint

        mismatches = verify_against_base(base, tv_dirs)
        if mismatches:
            print(
                "error: base fingerprint mismatch for: " + ", ".join(mismatches),
                file=sys.stderr,
            )
            return 3

    summary = merge_model(base, entries, settings, Path(out_str))

    meta = build_merged_metadata(
        base_model=str(base),
        task_vectors=[
            {"dir": str(e.dir), "kind": e.kind, "weight": e.weight} for e in entries
        ],
        base_fingerprint=summary.get("base_fingerprint"),
        merge_settings={
            "merge_method": settings.merge_method.value,
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
    )
    write_metadata(Path(out_str), meta)
    print(json.dumps({"out": str(Path(out_str)), **summary}, indent=2))
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
    pm.add_argument("--config", "-c", default=None, help="YAML recipe file (provides base, out, tv, model, and tuning params)")
    pm.add_argument("--base", default=None, help="base model dir or HF repo id (overrides YAML)")
    pm.add_argument("--out", default=None, help="output merged-model directory (overrides YAML)")
    pm.add_argument(
        "--merge-method",
        choices=[m.value for m in MergeMethod],
        default=None,
        help="merge method: gta (default) | linear | mixture | slerp | ties",
    )
    pm.add_argument("--tv", action="append", help="task-vector dir:weight (repeatable; overrides YAML)")
    pm.add_argument("--model", action="append", help="source model dir:weight (repeatable; overrides YAML)")
    pm.add_argument("--consensus", choices=["none", "sum", "count"], default=None)
    pm.add_argument("--density", type=float, default=None)
    pm.add_argument(
        "--method",
        choices=[m.value for m in SparsificationMethod],
        default=None,
    )
    pm.add_argument("--n", type=int, default=None, help="BS block top-n")
    pm.add_argument("--m", type=int, default=None, help="BS block size")
    pm.add_argument("--gamma", type=float, default=None)
    pm.add_argument("--epsilon", type=float, default=None)
    pm.add_argument("--no-rescale", action="store_true", default=None)
    pm.add_argument("--no-normalize", action="store_true", default=None)
    pm.add_argument("--lambda", dest="lambda_", type=float, default=None)
    pm.add_argument("--chunk-elements", type=int, default=None)
    pm.add_argument("--skip-fingerprint-check", action="store_true", default=None)
    pm.set_defaults(func=cmd_merge)

    pi = sub.add_parser("inspect", help="show metadata/manifest of a TV or merged model")
    pi.add_argument("--dir", required=True)
    pi.set_defaults(func=cmd_inspect)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
