# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Materialize legacy runtime selection state in the selected writable root."""

from __future__ import annotations

import os

from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)


def _read_validated_runtime_state(path: str) -> dict[str, object]:
    """Read and validate one canonical runtime-state document."""

    return validate_versioned_metadata_payload(
        MetadataKind.RUNTIME_STATE,
        read_yaml_mapping_file(path),
    )


def materialize_legacy_runtime_state(source_path: str, target_path: str) -> bool:
    """Copy valid legacy state once without modifying or replacing its source."""

    source = os.path.abspath(source_path)
    target = os.path.abspath(target_path)
    if source == target:
        if os.path.isfile(target):
            _read_validated_runtime_state(target)
        return False
    if os.path.exists(target):
        _read_validated_runtime_state(target)
        return False
    if not os.path.isfile(source):
        return False

    payload = _read_validated_runtime_state(source)
    atomic_write_validated_yaml_file(
        target,
        payload,
        validator=lambda candidate: validate_versioned_metadata_payload(
            MetadataKind.RUNTIME_STATE,
            candidate,
        ),
        verify_reader=_read_validated_runtime_state,
        smoke_check_reader=_read_validated_runtime_state,
    )
    return True
