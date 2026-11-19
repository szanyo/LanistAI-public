# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Offline CLI for explicit external-workspace migration operations."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lanista.reinforcengine.rl_model.storage.workspace_migration import (
    ExternalWorkspaceMigrationService,
)


def build_parser() -> argparse.ArgumentParser:
    """Build a parser whose every operation is read-only without ``--write``."""

    parser = argparse.ArgumentParser(
        description=(
            "Copy one legacy rl_model_storage workspace into an explicit LanistAI "
            "data home without changing the source."
        )
    )
    parser.add_argument(
        "--source-storage",
        required=True,
        help="Absolute path of the legacy rl_model_storage directory.",
    )
    parser.add_argument(
        "--destination-data-home",
        required=True,
        help="Absolute data home; the fixed target is DATA_HOME/rl_model_storage.",
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--resume",
        metavar="JOURNAL",
        help="Resume the exact root-confined migration journal.",
    )
    operation.add_argument(
        "--rollback",
        metavar="RECEIPT",
        help="Retain the published copy as verified recovery data.",
    )
    operation.add_argument(
        "--cleanup",
        metavar="RECEIPT",
        help="Remove verified migration controls or rollback recovery data.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the selected operation. The default is a read-only preview.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one explicit offline operation and return a process status code."""

    arguments = build_parser().parse_args(argv)
    service = ExternalWorkspaceMigrationService()
    try:
        if arguments.resume:
            result = service.resume(
                arguments.source_storage,
                arguments.destination_data_home,
                arguments.resume,
                dry_run=not arguments.write,
            )
        elif arguments.rollback:
            result = service.rollback(
                arguments.source_storage,
                arguments.destination_data_home,
                arguments.rollback,
                dry_run=not arguments.write,
            )
        elif arguments.cleanup:
            result = service.cleanup(
                arguments.source_storage,
                arguments.destination_data_home,
                arguments.cleanup,
                dry_run=not arguments.write,
            )
        else:
            result = service.migrate(
                arguments.source_storage,
                arguments.destination_data_home,
                dry_run=not arguments.write,
            )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result.to_mapping(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
