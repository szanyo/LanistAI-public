# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the tree nodes interface component."""

from typing import cast

from textual.widgets._tree import TreeNode


def first_data_node[NodeData](
    node: TreeNode[object],
    data_type: type[NodeData],
) -> TreeNode[NodeData] | None:
    """Return the first descendant whose data matches ``data_type``."""

    for child in node.children:
        if isinstance(child.data, data_type):
            return cast(TreeNode[NodeData], child)
        nested = first_data_node(cast(TreeNode[object], child), data_type)
        if nested is not None:
            return nested
    return None


def first_string_data_node[NodeData](
    node: TreeNode[NodeData],
) -> TreeNode[NodeData] | None:
    """Return the first descendant carrying string data in depth-first order."""
    for child in node.children:
        if isinstance(child.data, str):
            return child
        nested = first_string_data_node(child)
        if nested is not None:
            return nested
    return None
