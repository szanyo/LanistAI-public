# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral validation of the configured Ray logger hierarchy."""

from __future__ import annotations

import logging

from lanista.foundation.logging import LoggerInterface


class LoggerIntegrityError(RuntimeError):
    """Raised when Ray loggers violate the application logging contract."""


class LoggerIntegrityVerifier(LoggerInterface):
    """Verify propagation, handler ownership, and root-parent consistency."""

    def verify(self) -> None:
        """Verify the operation owned by this logger integrity verifier."""
        self._logger.debug("Logger integrity verification started.")
        violations = self._collect_ray_logger_violations()
        for violation in violations:
            self._logger.error(violation)
        if violations:
            raise LoggerIntegrityError(
                f"Ray logger integrity check found {len(violations)} violation(s)."
            )
        self._logger.debug("Logger integrity verification finished successfully.")

    def _collect_ray_logger_violations(self) -> list[str]:
        """Collect the ray logger violations owned by this logger integrity verifier."""
        violations: list[str] = []
        root_ray_logger = logging.getLogger("ray")
        self._append_logger_violations(
            logger=root_ray_logger,
            name="ray",
            violations=violations,
            check_parent=True,
        )
        for name in logging.root.manager.loggerDict:
            if name.startswith("ray."):
                self._append_logger_violations(
                    logger=logging.getLogger(name),
                    name=name,
                    violations=violations,
                    check_parent=False,
                )
        return violations

    def _append_logger_violations(
        self,
        *,
        logger: logging.Logger,
        name: str,
        violations: list[str],
        check_parent: bool,
    ) -> None:
        """Append the logger violations owned by this logger integrity verifier."""
        if not logger.propagate:
            violations.append(f"Ray logger does not propagate: {name}")
        if len(logger.handlers) > 1:
            violations.append(f"Ray logger has more than one handler: {name}")
        expected_parent = getattr(self._logger, "parent", None)
        if (
            check_parent
            and logger.parent is not None
            and expected_parent is not None
            and logger.parent.name != expected_parent.name
        ):
            violations.append(
                "Ray root logger has the wrong parent: "
                f"{logger.parent.name}; expected {expected_parent.name}"
            )
