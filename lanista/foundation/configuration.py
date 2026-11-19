# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Small YAML-backed configuration objects with nested attribute access."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml

_ConfigurationLoader: Any = type("_ConfigurationLoader", (yaml.SafeLoader,), {})
_ConfigurationDumper: Any = type("_ConfigurationDumper", (yaml.SafeDumper,), {})


def _construct_tuple(loader: Any, node: yaml.Node) -> tuple[object, ...]:
    """Return the construct tuple value owned by this configuration module."""
    return tuple(loader.construct_sequence(node, deep=True))


def _represent_tuple(
    dumper: Any,
    value: tuple[object, ...],
) -> yaml.Node:
    """Return the represent tuple value owned by this configuration module."""
    return dumper.represent_sequence("tag:yaml.org,2002:python/tuple", value)


_ConfigurationLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)
_ConfigurationDumper.add_representer(tuple, _represent_tuple)


@dataclass
class _ChangeState:
    """Represent runtime state for change."""

    changed: bool = False


class NestedDictWrapper:
    """Expose a nested mutable mapping through attributes and item access."""

    _data: dict[str, Any]
    _access_path: tuple[str, ...]
    _change_state: _ChangeState

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        access_path: tuple[str, ...] | list[str] | None = None,
        *,
        change_state: object | None = None,
    ) -> None:
        """Initialize the nested dict wrapper and its owned state."""
        object.__setattr__(self, "_data", data if data is not None else {})
        object.__setattr__(self, "_access_path", tuple(access_path or ()))
        object.__setattr__(
            self,
            "_change_state",
            change_state if isinstance(change_state, _ChangeState) else _ChangeState(),
        )

    def __getattr__(self, name: str) -> Any:
        """Resolve a delegated attribute exposed by this nested dict wrapper."""
        try:
            value = self._data[name]
        except KeyError as error:
            raise AttributeError(f"{self._display_path(name)!r} not found in configuration.") from error
        return self._wrap(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign an attribute while preserving this nested dict wrapper's invariants."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._set_value(name, value)

    def __delattr__(self, name: str) -> None:
        """Delete the operation owned by this nested dict wrapper."""
        try:
            del self._data[name]
        except KeyError as error:
            raise AttributeError(
                f"Cannot delete {self._display_path(name)!r}; it is not in configuration."
            ) from error
        self._change_state.changed = True

    def __getitem__(self, key: str) -> Any:
        """Return the requested value from this nested dict wrapper."""
        try:
            value = self._data[key]
        except KeyError as error:
            raise KeyError(f"{self._display_path(key)!r} not found in configuration.") from error
        return self._wrap(key, value)

    def __setitem__(self, key: str, value: Any) -> None:
        """Store the requested value in this nested dict wrapper."""
        self._set_value(key, value)

    def __delitem__(self, key: str) -> None:
        """Remove the requested value from this nested dict wrapper."""
        try:
            del self._data[key]
        except KeyError as error:
            raise KeyError(
                f"Cannot delete {self._display_path(key)!r}; it is not in configuration."
            ) from error
        self._change_state.changed = True

    def _display_path(self, name: str) -> str:
        """Return the display path value owned by this nested dict wrapper."""
        return ".".join((*self._access_path, name))

    def _wrap(self, name: str, value: Any) -> Any:
        """Return the wrap value owned by this nested dict wrapper."""
        if isinstance(value, dict):
            return NestedDictWrapper(
                value,
                (*self._access_path, name),
                change_state=self._change_state,
            )
        return value

    def _set_value(self, dotted_key: str, value: Any) -> None:
        """Set the value owned by this nested dict wrapper."""
        keys = dotted_key.split(".")
        target = self._data
        for key in keys[:-1]:
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise TypeError(f"Cannot assign below non-mapping key {self._display_path(key)!r}.")
            target = nested
        leaf = keys[-1]
        if leaf not in target or target[leaf] != value:
            target[leaf] = value
            self._change_state.changed = True

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return the named values exposed by this nested dict wrapper."""
        for key in self._data:
            yield key, self._wrap(key, self._data[key])

    def get_data(self) -> dict[str, Any]:
        """Return the data exposed by this nested dict wrapper."""
        return self._data

    def __getstate__(self) -> dict[str, object]:
        """Return serializable state for this nested dict wrapper."""
        return {
            "_data": self._data,
            "_access_path": self._access_path,
            "_changed": self._change_state.changed,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Restore this nested dict wrapper from serialized state."""
        data = state.get("_data", {})
        access_path = state.get("_access_path", ())
        if not isinstance(data, dict):
            data = {}
        if not isinstance(access_path, list | tuple):
            access_path = ()
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_access_path", tuple(str(item) for item in access_path))
        object.__setattr__(self, "_change_state", _ChangeState(bool(state.get("_changed", False))))


class Configuration:
    """Mutable YAML configuration with change-aware, atomic persistence."""

    _config_file_path: str | None
    _config_dict: dict[str, Any]
    _change_state: _ChangeState
    _clone: bool

    def __init__(
        self,
        clone: bool = False,
        config_dict: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the configuration and its owned state."""
        object.__setattr__(self, "_config_file_path", None)
        object.__setattr__(self, "_config_dict", dict(config_dict or {}))
        object.__setattr__(self, "_change_state", _ChangeState())
        object.__setattr__(self, "_clone", clone)

    @property
    def _changed(self) -> bool:
        """Return the changed exposed by this configuration."""
        return self._change_state.changed

    @_changed.setter
    def _changed(self, value: bool) -> None:
        """Handle the changed operation owned by this configuration."""
        self._change_state.changed = bool(value)

    def __getattr__(self, name: str) -> Any:
        """Resolve a delegated attribute exposed by this configuration."""
        if name.startswith("_"):
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}.")
        try:
            value = self._config_dict[name]
        except KeyError as error:
            location = self._config_file_path or "unknown"
            raise AttributeError(
                f"{type(self).__name__} has no configuration key {name!r} in {location}."
            ) from error
        return self._wrap(name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        """Assign an attribute while preserving this configuration's invariants."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self._config_dict or self._config_dict[name] != value:
            self._config_dict[name] = value
            self._change_state.changed = True

    def __delattr__(self, name: str) -> None:
        """Delete the operation owned by this configuration."""
        if name.startswith("_"):
            object.__delattr__(self, name)
            return
        try:
            del self._config_dict[name]
        except KeyError as error:
            raise AttributeError(f"Configuration key {name!r} does not exist.") from error
        self._change_state.changed = True

    def _wrap(self, name: str, value: Any) -> Any:
        """Return the wrap value owned by this configuration."""
        if isinstance(value, dict):
            return NestedDictWrapper(value, (name,), change_state=self._change_state)
        return value

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return the named values exposed by this configuration."""
        for key in self._config_dict:
            yield key, self._wrap(key, self._config_dict[key])

    def clone(self) -> Self:
        """Return an independent copy of this configuration."""
        return type(self)(clone=True, config_dict=deepcopy(self._config_dict))

    def get_config_file_path(self) -> str | None:
        """Return the config file path exposed by this configuration."""
        return self._config_file_path

    def set_config_file_path(self, path: str | os.PathLike[str]) -> None:
        """Set the config file path owned by this configuration."""
        if self._config_file_path is not None:
            raise ValueError("Configuration file path is already set.")
        object.__setattr__(self, "_config_file_path", os.fspath(path))

    def get_data(self) -> dict[str, Any]:
        """Return the data exposed by this configuration."""
        return self._config_dict

    def load(self) -> None:
        """Load and return the state owned by this configuration."""
        path = self._require_path()
        if not path.exists():
            self._change_state.changed = True
            self.save()
            return
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.load(stream, Loader=_ConfigurationLoader)
        if loaded is None:
            raise ValueError(f"Configuration file is empty: {path}")
        if not isinstance(loaded, Mapping):
            raise ValueError(f"Configuration root must be a mapping: {path}")
        object.__setattr__(self, "_config_dict", dict(loaded))
        self._change_state.changed = False

    def save(self) -> bool:
        """Persist the current state owned by this configuration."""
        path = self._require_path()
        if not self._change_state.changed:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.dump(
            self._config_dict,
            Dumper=_ConfigurationDumper,
            allow_unicode=True,
            sort_keys=False,
        )
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = stream.name
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        self._change_state.changed = False
        return True

    def _require_path(self) -> Path:
        """Return the required path for this configuration."""
        if self._config_file_path is None:
            raise ValueError("Configuration file path is not set.")
        return Path(self._config_file_path)

    def __getstate__(self) -> dict[str, object]:
        """Return serializable state for this configuration."""
        return {
            "_config_file_path": self._config_file_path,
            "_config_dict": self._config_dict,
            "_changed": self._change_state.changed,
            "_clone": self._clone,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Restore this configuration from serialized state."""
        path = state.get("_config_file_path")
        data = state.get("_config_dict", {})
        if isinstance(path, os.PathLike):
            path = os.fspath(path)
        elif not isinstance(path, str):
            path = None
        if not isinstance(data, dict):
            data = {}
        object.__setattr__(self, "_config_file_path", path)
        object.__setattr__(self, "_config_dict", data)
        object.__setattr__(self, "_change_state", _ChangeState(bool(state.get("_changed", False))))
        object.__setattr__(self, "_clone", bool(state.get("_clone", False)))


class AppConfiguration(Configuration):
    """Process-wide application configuration singleton."""

    _instance: Self | None = None

    def __new__(cls) -> Self:
        """Return the new value owned by this app configuration."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the app configuration and its owned state."""
        if getattr(self, "_app_configuration_initialized", False):
            return
        super().__init__()
        object.__setattr__(self, "_app_configuration_initialized", True)
