"""Run the existing easy configuration in memory and print three metrics."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import make_easy_cfg
from src.simulate import simulate_once


def main() -> None:
    result = simulate_once(make_easy_cfg(), 1, write_files=False)
    print(json.dumps({key: result["summary"][key] for key in
                      ("pdr", "delivered", "tx_attempts")}, sort_keys=True))


if __name__ == "__main__":
    main()
