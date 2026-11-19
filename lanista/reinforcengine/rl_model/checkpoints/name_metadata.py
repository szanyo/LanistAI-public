# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure metadata extraction from first-party checkpoint names."""

from __future__ import annotations

import re


def extract_checkpoint_steps(name: str) -> int:
    """Return the encoded step count, or ``-1`` when the name has none."""

    match = re.search(r"(?:_checkpoint_|_)([0-9]+(?:\.[0-9]+)?)_steps", name)
    if not match:
        return -1
    return int(float(match.group(1)))
