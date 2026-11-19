# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide a fail-closed atomic no-replacement directory publish primitive."""

from __future__ import annotations

import ctypes
import errno
import os

_RENAME_NOREPLACE = 1


class AtomicPublishUnavailable(RuntimeError):
    """Raised when the host cannot guarantee atomic no-replacement publish."""


def rename_directory_no_replace(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
) -> None:
    """Atomically rename one directory while refusing an existing target."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise AtomicPublishUnavailable(
            "The host does not provide renameat2(RENAME_NOREPLACE)."
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), destination_name)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise AtomicPublishUnavailable(
            "The filesystem does not support atomic no-replacement publish."
        )
    raise OSError(error_number, os.strerror(error_number), destination_name)


__all__ = ["AtomicPublishUnavailable", "rename_directory_no_replace"]
