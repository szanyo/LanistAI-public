# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Thread- and process-safe serialization for one storage root."""

from __future__ import annotations

import errno
import importlib
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, RLock, get_ident
from typing import Literal, Protocol, cast

from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard


class _FcntlModule(Protocol):
    """Define the typed boundary for fcntl module."""

    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> None:
        """Lock the operation owned by this fcntl module."""
        ...


class _MsvcrtModule(Protocol):
    """Define the typed boundary for msvcrt module."""

    LK_NBLCK: int
    LK_UNLCK: int

    def locking(self, descriptor: int, mode: int, byte_count: int) -> None:
        """Handle the locking operation owned by this msvcrt module."""
        ...


def _optional_platform_module[ModuleT](name: str) -> ModuleT | None:
    """Return the optional platform module value owned by this storage mutation lock module."""
    try:
        return cast(ModuleT, importlib.import_module(name))
    except ImportError:  # pragma: no cover - selected by the host platform
        return None


fcntl: _FcntlModule | None = _optional_platform_module("fcntl")
msvcrt: _MsvcrtModule | None = _optional_platform_module("msvcrt")


LOCK_FILE_NAME = ".storage-mutation.lock"


class StorageMutationLockError(RuntimeError):
    """Base error for a storage root that cannot be safely locked."""


class StorageMutationLockTimeout(StorageMutationLockError):
    """Raised when another mutation retains the root beyond the deadline."""


@dataclass(slots=True)
class _RootLockState:
    """Process-local ownership state for one canonical storage root."""

    thread_lock: RLock = field(default_factory=RLock)
    owner_thread_id: int | None = None
    descriptor: int | None = None
    depth: int = 0


_REGISTRY_GUARD = Lock()
_ROOT_LOCKS: dict[tuple[int, Path], _RootLockState] = {}


def _root_lock_state(root: Path) -> _RootLockState:
    # Including the PID prevents inherited pre-fork thread state from being
    # mistaken for reentrant ownership in the child process.
    """Return the root lock state value owned by this storage mutation lock module."""
    key = (os.getpid(), root)
    with _REGISTRY_GUARD:
        return _ROOT_LOCKS.setdefault(key, _RootLockState())


class StorageMutationLock:
    """Hold an exclusive lock across preflight, CAS, and durable commit.

    Separate lock instances are deliberately reentrant on the owning thread.
    Only the outermost instance owns the operating-system descriptor; other
    threads and processes remain excluded until that instance exits.
    """

    root: Path
    timeout_seconds: float = 5.0

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the storage mutation lock and its owned state."""
        guard = StoragePathGuard(root)
        if not guard.root.is_dir():
            raise StorageMutationLockError("Managed storage root is unavailable.")
        self.root = guard.root
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._state: _RootLockState | None = None
        self._entered = False

    def __enter__(self) -> StorageMutationLock:
        """Enter this storage mutation lock's managed context."""
        if self._entered:
            raise StorageMutationLockError(
                "The same storage lock instance cannot be entered twice."
            )

        deadline = time.monotonic() + self.timeout_seconds
        state = _root_lock_state(self.root)
        if not state.thread_lock.acquire(timeout=self.timeout_seconds):
            raise StorageMutationLockTimeout(
                "Another storage mutation is still in progress."
            )

        descriptor: int | None = None
        try:
            thread_id = get_ident()
            if state.owner_thread_id == thread_id:
                state.depth += 1
            else:
                if state.depth != 0 or state.descriptor is not None:
                    raise StorageMutationLockError(
                        "The process-local storage lock state is inconsistent."
                    )
                descriptor = self._open_lock_target()
                self._acquire_process_lock(descriptor, deadline)
                state.owner_thread_id = thread_id
                state.descriptor = descriptor
                state.depth = 1
                descriptor = None

            self._state = state
            self._entered = True
            return self
        except BaseException:
            try:
                if descriptor is not None:
                    os.close(descriptor)
            finally:
                state.thread_lock.release()
            raise

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> Literal[False]:
        """Release owned resources when leaving this storage mutation lock's managed context."""
        state = self._state
        if not self._entered or state is None:
            return False
        if state.owner_thread_id != get_ident():
            # Keep the context intact so its owning thread can still release it.
            raise StorageMutationLockError(
                "The storage mutation lock is owned by another thread."
            )

        self._entered = False
        self._state = None
        try:
            if state.depth <= 0:
                raise StorageMutationLockError(
                    "The storage mutation lock is not owned by this thread."
                )

            state.depth -= 1
            if state.depth == 0:
                descriptor = state.descriptor
                state.descriptor = None
                state.owner_thread_id = None
                if descriptor is not None:
                    # Closing must still be attempted if the explicit unlock
                    # reports an error; otherwise the lock can leak indefinitely.
                    try:
                        self._release_process_lock(descriptor)
                    finally:
                        os.close(descriptor)
        finally:
            state.thread_lock.release()
        return False

    def _open_lock_target(self) -> int:
        """Open the lock target workflow owned by this storage mutation lock."""
        if fcntl is not None:
            return self._open_posix_root_directory()
        if msvcrt is not None:  # pragma: no cover - Windows fallback
            return self._open_windows_lock_file()
        raise StorageMutationLockError(
            "Cross-process storage locking is unsupported."
        )

    def _open_posix_root_directory(self) -> int:
        """Open the root itself so POSIX locking leaves no lock artifact."""
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            raise StorageMutationLockError(
                "Safe POSIX directory locking is unsupported on this platform."
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= os.O_DIRECTORY | os.O_NOFOLLOW
        try:
            descriptor = os.open(self.root, flags)
        except OSError as exc:
            raise StorageMutationLockError(
                "The storage root cannot be opened safely for locking."
            ) from exc

        try:
            mode = os.fstat(descriptor).st_mode
        except OSError as exc:
            os.close(descriptor)
            raise StorageMutationLockError(
                "The storage lock target cannot be inspected safely."
            ) from exc
        if not stat.S_ISDIR(mode):
            os.close(descriptor)
            raise StorageMutationLockError(
                "The storage mutation lock target is not a directory."
            )
        return descriptor

    def _open_windows_lock_file(self) -> int:  # pragma: no cover - Windows fallback
        """Open the windows lock file workflow owned by this storage mutation lock."""
        lock_path = self.root / LOCK_FILE_NAME
        try:
            if os.path.lexists(lock_path) and stat.S_ISLNK(lock_path.lstat().st_mode):
                raise StorageMutationLockError(
                    "The storage mutation lock target cannot be a symbolic link."
                )
        except OSError as exc:
            raise StorageMutationLockError(
                "The storage mutation lock target cannot be inspected safely."
            ) from exc

        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOINHERIT", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise StorageMutationLockError(
                "The storage mutation lock file cannot be opened safely."
            ) from exc

        try:
            descriptor_stat = os.fstat(descriptor)
            path_stat = lock_path.lstat()
            if (
                not stat.S_ISREG(descriptor_stat.st_mode)
                or stat.S_ISLNK(path_stat.st_mode)
                or (descriptor_stat.st_dev, descriptor_stat.st_ino)
                != (path_stat.st_dev, path_stat.st_ino)
            ):
                raise StorageMutationLockError(
                    "The storage mutation lock target is not a safe regular file."
                )
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def _acquire_process_lock(self, descriptor: int, deadline: float) -> None:
        """Acquire the process lock owned by this storage mutation lock."""
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                elif msvcrt is not None:  # pragma: no cover - Windows fallback
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    if os.fstat(descriptor).st_size == 0:
                        os.write(descriptor, b"\0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - guarded by _open_lock_target
                    raise StorageMutationLockError(
                        "Cross-process storage locking is unsupported."
                    )
                return
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise StorageMutationLockError(
                        "The storage mutation lock could not be acquired."
                    ) from exc
                if time.monotonic() >= deadline:
                    raise StorageMutationLockTimeout(
                        "Another storage mutation is still in progress."
                    ) from exc
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _release_process_lock(descriptor: int) -> None:
        """Release the process lock owned by this storage mutation lock."""
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - Windows fallback
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        except OSError as exc:
            raise StorageMutationLockError(
                "The storage mutation lock could not be released."
            ) from exc
