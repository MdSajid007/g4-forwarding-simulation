"""Check the frozen G4d rows, configuration bindings, summaries, and report."""
from __future__ import annotations

import csv
import dataclasses
import json
import math
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual.g4d_oracle_gap import (
    DEFAULT_SEEDS,
    G4dPairedResult,
    evaluate_g4d_verdict,
    load_g4d_scenarios,
    render_report,
    summarize_by_scenario,
)
from src.hashing import cfg_hash


def load_rows(path: Path) -> list[G4dPairedResult]:
    hints = typing.get_type_hints(G4dPairedResult)

    def convert(key: str, value: str) -> object:
        kind = hints[key]
        if kind is bool:
            if value not in ("True", "False"):
                raise ValueError(f"Invalid boolean for {key}: {value}")
            return value == "True"
        if kind is int:
            return int(value)
        if kind is float:
            return float(value)
        if typing.get_origin(kind) is typing.Union:
            return None if value == "" else float(value)
        return value

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != set(hints):
            raise ValueError("Paired-result columns do not match the preserved schema")
        return [G4dPairedResult(**{k: convert(k, v) for k, v in row.items()})
                for row in reader]


def verify_results() -> dict[str, object]:
    rows = load_rows(ROOT / "outputs/g4d_mode_comparison_results.csv")
    scenarios = load_g4d_scenarios()
    expected_pairs = {(name, seed) for name in scenarios for seed in DEFAULT_SEEDS}
    actual_pairs = [(row.scenario_name, row.seed) for row in rows]
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise ValueError("Expected exactly nine scenarios by ten seeds without duplicates")
    for row in rows:
        if row.cfg_hash != cfg_hash(scenarios[row.scenario_name]):
            raise ValueError(f"Configuration mismatch: {row.scenario_name}, seed {row.seed}")
        if not row.pairing_checks_passed:
            raise ValueError("A recorded initialization-pairing check failed")

    summaries = summarize_by_scenario(rows)
    with (ROOT / "outputs/g4d_oracle_gap_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        expected_rows = list(csv.DictReader(stream))
    expected = {row["scenario_name"]: row for row in expected_rows}
    if len(expected_rows) != len(scenarios) or set(expected) != set(scenarios):
        raise ValueError("Unexpected summary scenario coverage")
    for summary in summaries:
        for key, value in dataclasses.asdict(summary).items():
            recorded = expected[summary.scenario_name][key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                matches = math.isclose(value, float(recorded), rel_tol=1e-12, abs_tol=1e-12)
            else:
                matches = str(value) == recorded
            if not matches:
                raise ValueError(f"Summary mismatch: {summary.scenario_name}/{key}")

    verdict = evaluate_g4d_verdict(summaries)
    report = render_report(summaries, verdict, seeds=DEFAULT_SEEDS)
    if report != (ROOT / "docs/g4d_oracle_gap_report.md").read_text(encoding="utf-8"):
        raise ValueError("Recomputed report text differs from the frozen report")
    return {
        "verified": True,
        "paired_rows": len(rows),
        "scenarios": len(summaries),
        "verdict": verdict.verdict,
        "g5_authorized": verdict.g5_authorized,
    }


if __name__ == "__main__":
    print(json.dumps(verify_results(), sort_keys=True))
