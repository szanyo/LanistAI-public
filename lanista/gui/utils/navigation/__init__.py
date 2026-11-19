# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Navigation package marker.

Consumers import the coordinator and graph modules explicitly. Keeping package
initialization side-effect free prevents the package/module import cycle that
otherwise occurs while the graph loads route identifiers.
"""
