# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide offline regeneration of one evaluation-library report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lanista.reinforcengine.rl_metrics.evaluation_report import (
    GROUP_SUMMARY_FILENAME,
    GROUPED_LIBRARY_SUMMARY_FILENAME,
    GRID_SUMMARY_FILENAME,
    LIBRARY_SUMMARY_FILENAME,
    SCENARIO_SUMMARY_FILENAME,
    EvaluationReportService,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the explicit run-root parser without consulting runtime state."""

    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the stable CSV/JSON publication report for one terminal "
            "LanistAI evaluation-library sweep."
        )
    )
    parser.add_argument(
        "run_root",
        help="Sweep directory containing evaluation-library-run.json and scenarios/.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one report and return a process-compatible status code."""

    arguments = build_parser().parse_args(argv)
    try:
        result = EvaluationReportService().generate(arguments.run_root)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2

    payload = {
        "status": "completed",
        "summary_root": str(result.summary_root),
        "scenario_count": len(result.scenarios),
        "grid_cell_count": len(result.grid),
        "artifacts": [
            SCENARIO_SUMMARY_FILENAME,
            GRID_SUMMARY_FILENAME,
            LIBRARY_SUMMARY_FILENAME,
        ],
    }
    if getattr(result, "library", None) is not None:
        payload["group_count"] = len(result.groups)
        payload["artifacts"] = [
            SCENARIO_SUMMARY_FILENAME,
            GROUP_SUMMARY_FILENAME,
            GRID_SUMMARY_FILENAME,
            GROUPED_LIBRARY_SUMMARY_FILENAME,
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
