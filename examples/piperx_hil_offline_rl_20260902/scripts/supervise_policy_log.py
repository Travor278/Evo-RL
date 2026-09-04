#!/usr/bin/env python
"""Conservative ETA and numerical-health gate for policy training logs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


METRIC_PATTERN = re.compile(
    r"step:(\d+)\s+smpl:[\s\S]{0,140}?"
    r"loss:(\S+)\s+grdn:(\S+)\s+lr:(\S+)\s+"
    r"updt_s:(\S+)\s+data_s:(\S+)"
)

FATAL_PATTERNS = {
    "non_finite": re.compile(r"Non-finite|\bnan\b|\binf\b", re.IGNORECASE),
    "cuda": re.compile(r"CUDA error|OutOfMemoryError|CUDA out of memory", re.IGNORECASE),
    "nccl": re.compile(r"NCCL[^\n]*(?:error|unhandled|failed)", re.IGNORECASE),
    "ddp": re.compile(r"DistributedDataParallel[^\n]*(?:error|failed)|Expected to have finished reduction"),
    "decode": re.compile(r"FrameTimestampError|video decode[^\n]*(?:error|failed)", re.IGNORECASE),
    "checkpoint": re.compile(r"checkpoint[^\n]*(?:error|failed|failure)", re.IGNORECASE),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--target-steps", type=int, default=10_000)
    parser.add_argument("--max-projected-seconds", type=float, default=216_000)
    parser.add_argument("--min-metrics", type=int, default=2)
    return parser.parse_args()


def number(raw: str) -> float:
    try:
        return float(raw.rstrip(","))
    except ValueError:
        return math.nan


def main() -> int:
    args = parse_args()
    text = args.log.read_text(errors="replace") if args.log.exists() else ""
    lines = text.splitlines()
    boundary_pattern = re.compile(
        rf"^(?:\[[^\]]+\])*\s*{re.escape(args.marker)}(?:\s|$)"
    )
    boundary_indices = [index for index, line in enumerate(lines) if boundary_pattern.search(line)]
    segment = "\n".join(lines[boundary_indices[-1] + 1 :]) if boundary_indices else ""
    fatal_hits = [name for name, pattern in FATAL_PATTERNS.items() if pattern.search(segment)]

    metrics_by_step: dict[int, dict[str, float | int]] = {}
    for match in METRIC_PATTERN.finditer(segment):
        step = int(match.group(1))
        loss, grad_norm, learning_rate, update_s, data_s = map(number, match.groups()[1:])
        total_s = update_s + data_s
        metrics_by_step[step] = {
            "step": step,
            "loss": loss,
            "grad_norm": grad_norm,
            "learning_rate": learning_rate,
            "update_s": update_s,
            "data_s": data_s,
            "total_s": total_s,
            "projected_seconds": total_s * args.target_steps,
        }

    metrics = sorted(metrics_by_step.values(), key=lambda item: int(item["step"]))
    recent = metrics[-args.min_metrics :]
    numerical_bad = [
        metric
        for metric in metrics
        if not all(
            math.isfinite(float(metric[key]))
            for key in ("loss", "grad_norm", "learning_rate", "update_s", "data_s")
        )
        or abs(float(metric["loss"])) > 100.0
        or abs(float(metric["grad_norm"])) > 1000.0
    ]
    slow = len(recent) >= args.min_metrics and all(
        float(metric["projected_seconds"]) > args.max_projected_seconds for metric in recent
    )

    decision = "continue"
    exit_code = 0
    if fatal_hits or numerical_bad:
        decision = "stop_numerical_or_runtime_error"
        exit_code = 21
    elif slow:
        decision = "stop_eta_gate"
        exit_code = 20

    serializable_recent = [
        {
            key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in metric.items()
        }
        for metric in recent
    ]
    payload = {
        "decision": decision,
        "marker": args.marker,
        "fatal_hits": fatal_hits,
        "metric_count": len(metrics),
        "recent_metrics": serializable_recent,
        "max_projected_seconds": args.max_projected_seconds,
        "target_steps": args.target_steps,
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
