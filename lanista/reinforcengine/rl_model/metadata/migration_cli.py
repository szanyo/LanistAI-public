# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Command-line entry point for explicit metadata migration and rollback."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from lanista.reinforcengine.rl_model.metadata.migration import MetadataSchemaMigrator


def build_parser() -> argparse.ArgumentParser:
    """Build a dry-run-first CLI parser without consulting application state."""
    parser = argparse.ArgumentParser(description="Migrate LanistAI storage metadata to schema version 1.")
    parser.add_argument("root", help="Explicit model or models storage root.")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--inventory", action="store_true", help="List known metadata files without writing.")
    operation.add_argument("--rollback", metavar="RECEIPT", help="Restore metadata recorded by a migration receipt.")
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply the migration or rollback. Without this flag every operation is read-only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute an explicit command and return a process-compatible status code."""
    arguments = build_parser().parse_args(argv)
    migrator = MetadataSchemaMigrator()
    try:
        if arguments.inventory:
            entries = migrator.inventory(arguments.root)
            payload = {
                "dry_run": True,
                "inventory_count": len(entries),
                "entries": [
                    {
                        "relative_path": entry.relative_path,
                        "kind": entry.kind.value,
                        "byte_size": entry.byte_size,
                        "sha256": entry.sha256,
                    }
                    for entry in entries
                ],
            }
        elif arguments.rollback:
            receipt_path = Path(arguments.rollback)
            result = migrator.rollback(
                arguments.root,
                receipt_path,
                dry_run=not arguments.write,
            )
            payload = result.to_mapping()
        else:
            result = migrator.migrate(arguments.root, dry_run=not arguments.write)
            payload = result.to_mapping()
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
