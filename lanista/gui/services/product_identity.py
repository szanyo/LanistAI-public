# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Canonical product identity shared by GUI presentation surfaces."""

import time

PRODUCT_NAME = "LanistAI"
PROJECT_DESCRIPTION = (
    "A modular, extensible, domain-independent reinforcement learning framework for reproducible\n"
    "training, evaluation, inference, and experiment comparison across diverse environments and "
    "application domains."
)
PRODUCT_VERSION = "v1.0.0"
CREATORS = ("Benedek Szanyó, BSc",)
RESEARCH_INSTITUTE = "HUN-REN Institute for Computer Science and Control (HUN-REN SZTAKI)"
COPYRIGHT_HOLDER = "HUN-REN SZTAKI"
DEPARTMENT = "Systems and Control Laboratory (SCL)"
RESEARCH_ADVISORS = ("Balázs Németh, DSc", "Dániel Fényes, PhD")
CONTRIBUTORS = ("Bence Jekl, MSc — provided the initial foundation for the project.",)
SZTAKI_LOGO = "\n".join(
    line.ljust(20)
    for line in (
        "     +-+++---+-",
        "   -++-+     -++-",
        "  -+            ++",
        " -+         -+----+",
        "-+         +-",
        "-+       -++      -+",
        "       -+-        ++",
        " +----++         +-",
        "  ++            -+",
        "   ----       --+",
        "     --+---+--+",
    )
)
PROJECT_START_YEAR = 2024
PROJECT_CREATION_YEARS = "2024-2026"
CURRENT_YEAR = time.localtime().tm_year
COPYRIGHT_NOTICE = (
    f"Copyright (c) {PROJECT_START_YEAR}-{CURRENT_YEAR} {COPYRIGHT_HOLDER}.\n"
    "All rights reserved."
)
