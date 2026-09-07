#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from lerobot.rl.value_failure_loo import aggregate_failure_loo, build_failure_loo_folds, folds_to_dicts


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build(args: argparse.Namespace) -> None:
    records = _read_json(args.outcomes)
    folds = build_failure_loo_folds(
        records,
        seed=args.seed,
        group_field=args.group_field,
        expected_failures=args.expected_failures,
    )
    payload = {
        "schema_version": "attempt-aware-value-failure-loo/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "group_field": args.group_field,
        "expected_failures": args.expected_failures,
        "matching_uses_model_predictions": False,
        "folds": folds_to_dicts(folds),
    }
    _write_json(args.output, payload)


def _aggregate(args: argparse.Namespace) -> None:
    fold_results = _read_json(args.fold_results)
    report = aggregate_failure_loo(
        fold_results,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        minimum_positive_folds=args.minimum_positive_folds,
    )
    payload = {
        "schema_version": "attempt-aware-value-failure-loo-gate/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "minimum_positive_folds": args.minimum_positive_folds,
        **report,
    }
    _write_json(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or aggregate grouped failure leave-one-out gates.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--outcomes", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=20260906)
    build.add_argument("--group-field", default="source_collection_id")
    build.add_argument("--expected-failures", type=int, default=7)
    build.set_defaults(func=_build)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--fold-results", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--seed", type=int, default=20260906)
    aggregate.add_argument("--bootstrap-samples", type=int, default=10_000)
    aggregate.add_argument("--minimum-positive-folds", type=int, default=5)
    aggregate.set_defaults(func=_aggregate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
