# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Simulink-style saturation helper."""


def saturation_block(
    x: float,
    lower_limit: float = -3,
    upper_limit: float = 2,
    detect_zero_crossing: bool = False,
    round_floor: bool = True,
) -> float:
    """
    This function is a Python implementation of the Simulink saturation block.

    :param x: The input signal.
    :param lower_limit: The lower saturation limit.
    :param upper_limit: The upper saturation limit.
    :param detect_zero_crossing: If True, the function detects zero crossing.
    :param round_floor: If True, the function applies floor rounding.
    :return: The saturated output signal.
    """
    # Apply floor rounding if specified (default)
    new_x: float | int = int(x // 1) if round_floor else x

    # Detect zero crossing if specified
    if detect_zero_crossing and (
        (new_x < lower_limit <= new_x + 1)
        or (new_x > upper_limit >= new_x - 1)
    ):
        print("Zero crossing detected")

    # Saturation logic
    if new_x < lower_limit:
        return lower_limit
    elif new_x > upper_limit:
        return upper_limit
    else:
        return x  # In Matlab the output data type is the same as the input data type
