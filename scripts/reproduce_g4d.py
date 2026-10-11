"""Replay the preserved G4d experiment without overwriting frozen results."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.counterfactual.g4d_oracle_gap import run_g4d_probe, write_g4d_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="reproduction/latest")
    args = parser.parse_args()
    destination = (ROOT / args.output_dir).resolve()
    allowed = (ROOT / "reproduction").resolve()
    if destination != allowed and allowed not in destination.parents:
        parser.error("--output-dir must be inside this checkout's reproduction directory")

    rows, summaries, verdict = run_g4d_probe()
    report, paired, summary = write_g4d_outputs(
        rows, summaries, verdict,
        docs_dir=destination / "docs",
        outputs_dir=destination / "outputs",
    )
    comparisons = {
        "paired_csv": paired.read_bytes() == (ROOT / "outputs" / paired.name).read_bytes(),
        "summary_csv": summary.read_bytes() == (ROOT / "outputs" / summary.name).read_bytes(),
        "report_text": report.read_text(encoding="utf-8") ==
                       (ROOT / "docs" / report.name).read_text(encoding="utf-8"),
    }
    matches = all(comparisons.values())
    print(json.dumps({
        "paired_rows": len(rows),
        "scenarios": len(summaries),
        "verdict": verdict.verdict,
        "g5_authorized": verdict.g5_authorized,
        "reference_matches": comparisons,
        "reproduced": matches,
        "output_dir": destination.relative_to(ROOT).as_posix(),
    }, sort_keys=True))
    return 0 if matches else 2


if __name__ == "__main__":
    raise SystemExit(main())
