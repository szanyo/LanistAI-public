# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed longitudinal-control optimization shared by the model plugins."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from itertools import product
from math import isfinite, sqrt

import numpy as np
from cvxopt import matrix, solvers
from cvxopt.solvers import options

VEHICLE_AVOIDANCE_RADIUS_M = 18.0
PEDESTRIAN_AVOIDANCE_RADIUS_M = 5.0
VEHICLE_CONTROL_HORIZON_S = 1.0
PEDESTRIAN_CONTROL_HORIZON_S = 0.5
CONSTRAINT_ERROR_MARGIN = 0.005
PEDESTRIAN_CONSTRAINT_COUNT = 10
OPTIMIZATION_LOWER_BOUND_MPS2 = -500.0
OPTIMIZATION_UPPER_BOUND_MPS2 = 500.0

options["show_progress"] = False


class PedestrianCrossingState(IntEnum):
    """Three-state pedestrian context consumed by the supervisor."""

    ON_CROSSING = 0
    CLEAR = 1
    NOT_APPLICABLE = 2


class QuadraticSolverBackend(StrEnum):
    """Supported scalar QP representations."""

    CVXOPT_MATRICES = "cvxopt"
    CVXOPT_NUMPY = "cvxopt_numpy"


@dataclass(frozen=True, slots=True)
class PedestrianConstraint:
    """Pedestrian station, speed, and crossing state with explicit units."""

    station_m: float
    speed_mps: float
    state: PedestrianCrossingState

    @classmethod
    def from_runtime_values(cls, values: Sequence[float]) -> PedestrianConstraint:
        """Build the typed context from the three simulator values."""

        if len(values) != 3:
            raise ValueError("pedestrian parameters must contain station, speed, and state")
        return cls(
            station_m=float(values[0]),
            speed_mps=float(values[1]),
            state=PedestrianCrossingState(int(values[2])),
        )


@dataclass(frozen=True, slots=True)
class LongitudinalControlRequest:
    """Complete input for one scalar longitudinal-control decision."""

    ego_station_m: float
    object_stations_m: tuple[float, float, float]
    ego_speed_mps: float
    object_speeds_mps: tuple[float, float, float]
    previous_command_mps2: float
    policy_command_mps2: float
    pedestrian: PedestrianConstraint

    @classmethod
    def from_runtime_values(
        cls,
        *,
        ego_station_m: float,
        object_stations_m: Sequence[float],
        ego_speed_mps: float,
        object_speeds_mps: Sequence[float],
        previous_command_mps2: float,
        policy_command_mps2: float,
        pedestrian_parameters: Sequence[float],
    ) -> LongitudinalControlRequest:
        """Validate mutable runtime sequences at the typed component boundary."""

        return cls(
            ego_station_m=float(ego_station_m),
            object_stations_m=_three_floats("object stations", object_stations_m),
            ego_speed_mps=float(ego_speed_mps),
            object_speeds_mps=_three_floats("object speeds", object_speeds_mps),
            previous_command_mps2=float(previous_command_mps2),
            policy_command_mps2=float(policy_command_mps2),
            pedestrian=PedestrianConstraint.from_runtime_values(pedestrian_parameters),
        )


@dataclass(frozen=True, slots=True)
class LongitudinalControlResult:
    """Optimizer delta and objective cost for one control step."""

    delta_mps2: float
    cost: float


@dataclass(frozen=True, slots=True)
class ScalarQuadraticProgram:
    """One-variable quadratic program accepted by both cvxopt adapters."""

    hessian: float
    linear_term: float
    inequality_coefficients: tuple[float, ...]
    inequality_bounds: tuple[float, ...]
    lower_bound: float = OPTIMIZATION_LOWER_BOUND_MPS2
    upper_bound: float = OPTIMIZATION_UPPER_BOUND_MPS2

    def __post_init__(self) -> None:
        """Validate and finalize the initialized scalar quadratic program state."""
        if len(self.inequality_coefficients) != len(self.inequality_bounds):
            raise ValueError("inequality coefficient and bound counts must match")


class CvxoptScalarQuadraticSolver:
    """Adapter from the typed scalar problem to cvxopt representations."""

    def solve(
        self,
        problem: ScalarQuadraticProgram,
        backend: QuadraticSolverBackend = QuadraticSolverBackend.CVXOPT_MATRICES,
    ) -> list[float]:
        """Solve the problem with the selected representation."""

        if backend is QuadraticSolverBackend.CVXOPT_NUMPY:
            return self.solve_numpy(problem)
        return self.solve_matrices(problem)

    @staticmethod
    def solve_numpy(problem: ScalarQuadraticProgram) -> list[float]:
        """Solve through NumPy-backed cvxopt matrices."""

        hessian = matrix(problem.hessian, (1, 1), "d")
        linear = matrix(problem.linear_term, (1, 1), "d")
        coefficients = matrix(
            problem.inequality_coefficients,
            (len(problem.inequality_coefficients), 1),
            "d",
        )
        bounds = matrix(
            problem.inequality_bounds,
            (len(problem.inequality_bounds), 1),
            "d",
        )
        lower = matrix(problem.lower_bound, (1, 1), "d")
        upper = matrix(problem.upper_bound, (1, 1), "d")
        inequalities = matrix(
            np.vstack((np.array(coefficients), np.eye(1), -np.eye(1)))
        )
        inequality_bounds = matrix(
            np.vstack((np.array(bounds), np.array(upper), np.array(-lower)))
        )
        solution = solvers.qp(hessian, linear, inequalities, inequality_bounds)
        return [float(value) for value in solution["x"]]

    @staticmethod
    def solve_matrices(problem: ScalarQuadraticProgram) -> list[float]:
        """Solve using native cvxopt matrices without NumPy conversion."""

        hessian = matrix(problem.hessian, (1, 1), "d")
        linear = matrix(problem.linear_term, (1, 1), "d")
        coefficients = matrix(
            problem.inequality_coefficients,
            (len(problem.inequality_coefficients), 1),
            "d",
        )
        bounds = matrix(
            problem.inequality_bounds,
            (len(problem.inequality_bounds), 1),
            "d",
        )
        lower = matrix(problem.lower_bound, (1, 1), "d")
        upper = matrix(problem.upper_bound, (1, 1), "d")
        inequalities = matrix(
            [coefficients[index] for index in range(coefficients.size[0])]
            + [-matrix(1.0, (1, 1))]
            + [matrix(1.0, (1, 1))]
        )
        inequality_bounds = matrix(
            [bounds[index] for index in range(bounds.size[0])]
            + [-lower[0]]
            + [upper[0]]
        )
        solution = solvers.qp(hessian, linear, inequalities, inequality_bounds)
        return [float(value) for value in solution["x"]]


@dataclass(frozen=True, slots=True)
class _CandidateResult:
    """Report the result of candidate."""

    delta_mps2: float
    cost: float


class LongitudinalControlOptimizer:
    """Calculate the collision-constrained scalar acceleration delta."""

    def __init__(
        self,
        solver: CvxoptScalarQuadraticSolver | None = None,
        *,
        backend: QuadraticSolverBackend = QuadraticSolverBackend.CVXOPT_MATRICES,
    ) -> None:
        """Initialize the longitudinal control optimizer and its owned state."""
        self._solver = solver or CvxoptScalarQuadraticSolver()
        self._backend = backend

    def optimize(self, request: LongitudinalControlRequest) -> LongitudinalControlResult:
        """Return the minimum-cost result without changing runtime state."""

        pedestrian_constraints: tuple[tuple[float, float], ...] = ()
        if request.pedestrian.state is PedestrianCrossingState.NOT_APPLICABLE:
            horizon_s = VEHICLE_CONTROL_HORIZON_S
            minimum_command_mps2 = -1.5
        else:
            horizon_s = PEDESTRIAN_CONTROL_HORIZON_S
            minimum_command_mps2 = -3.0
            pedestrian_constraints = self._pedestrian_constraints(request, horizon_s)

        candidates = [
            self._solve_candidate(
                request,
                horizon_s=horizon_s,
                vehicle_side=signs[0],
                minimum_command_mps2=minimum_command_mps2,
                pedestrian_constraints=pedestrian_constraints,
            )
            for signs in product((-1, 1), repeat=3)
        ]
        selected = min(candidates, key=lambda candidate: candidate.cost)
        return LongitudinalControlResult(selected.delta_mps2, selected.cost)

    def _pedestrian_constraints(
        self,
        request: LongitudinalControlRequest,
        horizon_s: float,
    ) -> tuple[tuple[float, float], ...]:
        """Return the pedestrian constraints value owned by this longitudinal control optimizer."""
        if request.pedestrian.state is PedestrianCrossingState.CLEAR:
            return ()
        constraints: list[tuple[float, float]] = []
        for _ in range(PEDESTRIAN_CONSTRAINT_COUNT):
            try:
                constraint = _circle_avoidance_constraint(
                    ego_station_m=request.ego_station_m,
                    object_station_m=request.pedestrian.station_m,
                    ego_speed_mps=request.ego_speed_mps,
                    object_speed_mps=request.pedestrian.speed_mps,
                    horizon_s=horizon_s,
                    previous_command_mps2=request.previous_command_mps2,
                    policy_command_mps2=request.policy_command_mps2,
                    side=1,
                    radius_m=PEDESTRIAN_AVOIDANCE_RADIUS_M,
                )
            except Exception:
                # Malformed active geometry must not silently remove braking.
                constraint = (
                    0.5 * horizon_s**2,
                    -request.ego_speed_mps * horizon_s
                    - 0.5 * request.previous_command_mps2 * horizon_s**2,
                )
            constraints.append(constraint)
        return tuple(constraints)

    def _solve_candidate(
        self,
        request: LongitudinalControlRequest,
        *,
        horizon_s: float,
        vehicle_side: int,
        minimum_command_mps2: float,
        pedestrian_constraints: tuple[tuple[float, float], ...],
    ) -> _CandidateResult:
        """Return the solve candidate value owned by this longitudinal control optimizer."""
        try:
            vehicle_constraints = self._vehicle_constraints(
                request,
                horizon_s=horizon_s,
                side=vehicle_side,
            )
            rollback_constraint = (
                -0.5 * horizon_s**2,
                request.ego_speed_mps * horizon_s
                + 0.5 * request.previous_command_mps2 * horizon_s**2,
            )
            constraints = (rollback_constraint, *vehicle_constraints, *pedestrian_constraints)
            problem = ScalarQuadraticProgram(
                hessian=1.0,
                linear_term=request.previous_command_mps2 - request.policy_command_mps2,
                inequality_coefficients=tuple(item[0] for item in constraints),
                inequality_bounds=tuple(item[1] for item in constraints),
            )
            solution = self._solver.solve(problem, self._backend)
            if solution:
                return _CandidateResult(
                    delta_mps2=solution[0],
                    cost=_objective_cost(solution, request),
                )
            return _CandidateResult(problem.lower_bound, 2e25)
        except Exception:
            # QP infeasibility/domain failures use the established emergency
            # deceleration candidate and sentinel cost.
            return _CandidateResult(
                minimum_command_mps2 - request.previous_command_mps2,
                5e25,
            )

    @staticmethod
    def _vehicle_constraints(
        request: LongitudinalControlRequest,
        *,
        horizon_s: float,
        side: int,
    ) -> tuple[tuple[float, float], ...]:
        """Return the vehicle constraints value owned by this longitudinal control optimizer."""
        return tuple(
            _circle_avoidance_constraint(
                ego_station_m=request.ego_station_m,
                object_station_m=station_m,
                ego_speed_mps=request.ego_speed_mps,
                object_speed_mps=speed_mps,
                horizon_s=horizon_s,
                previous_command_mps2=request.previous_command_mps2,
                policy_command_mps2=request.policy_command_mps2,
                side=side,
                radius_m=VEHICLE_AVOIDANCE_RADIUS_M,
            )
            for station_m, speed_mps in zip(
                request.object_stations_m,
                request.object_speeds_mps,
                strict=True,
            )
            if _vehicle_constraint_is_active(
                ego_station_m=request.ego_station_m,
                object_station_m=station_m,
                ego_speed_mps=request.ego_speed_mps,
                object_speed_mps=speed_mps,
                previous_command_mps2=request.previous_command_mps2,
                policy_command_mps2=request.policy_command_mps2,
                horizon_s=horizon_s,
            )
        )


def nominal_vehicle_avoidance_approach_station_m(
    object_station_m: float,
    ego_speed_mps: float = 0.0,
    *,
    horizon_s: float = VEHICLE_CONTROL_HORIZON_S,
    radius_m: float = VEHICLE_AVOIDANCE_RADIUS_M,
) -> float:
    """Return the negative-branch nominal QP approach boundary.

    The reference deliberately excludes solver error margins, acceleration
    commands, and pedestrian constraints.  It is the calibrated vehicle-circle
    geometry projected over the normal vehicle-control horizon.
    """

    object_station = float(object_station_m)
    speed = float(ego_speed_mps)
    horizon = float(horizon_s)
    radius = float(radius_m)
    if not all(isfinite(value) for value in (object_station, speed, horizon, radius)):
        raise ValueError("vehicle-avoidance reference inputs must be finite")
    if radius <= 0.0:
        raise ValueError("vehicle-avoidance radius must be positive")
    if horizon < 0.0:
        raise ValueError("vehicle-control horizon must be non-negative")
    stationary_boundary = _nominal_circle_approach_station_m(
        object_station,
        radius,
        context="vehicle-avoidance",
    )
    return stationary_boundary - (max(0.0, speed) * horizon)


def nominal_pedestrian_avoidance_approach_station_m(
    object_station_m: float,
    *,
    radius_m: float = PEDESTRIAN_AVOIDANCE_RADIUS_M,
) -> float:
    """Return the stationary negative-branch boundary of the pedestrian QP circle."""

    return _nominal_circle_approach_station_m(
        object_station_m,
        radius_m,
        context="pedestrian-avoidance",
    )


def _nominal_circle_approach_station_m(
    object_station_m: float,
    radius_m: float,
    *,
    context: str,
) -> float:
    """Return the negative ego-station branch of one circular constraint."""

    object_station = float(object_station_m)
    radius = float(radius_m)
    if not all(isfinite(value) for value in (object_station, radius)):
        raise ValueError(f"{context} reference inputs must be finite")
    if radius <= 0.0:
        raise ValueError(f"{context} radius must be positive")
    if abs(object_station) > radius:
        raise ValueError(f"object station lies outside the {context} circle")
    return -sqrt(max(0.0, radius**2 - object_station**2))


def _vehicle_constraint_is_active(
    *,
    ego_station_m: float,
    object_station_m: float,
    ego_speed_mps: float,
    object_speed_mps: float,
    previous_command_mps2: float,
    policy_command_mps2: float,
    horizon_s: float,
    radius_m: float = VEHICLE_AVOIDANCE_RADIUS_M,
) -> bool:
    """Return whether an actor can enter the QP conflict circle this horizon."""

    values = (
        ego_station_m,
        object_station_m,
        ego_speed_mps,
        object_speed_mps,
        previous_command_mps2,
        policy_command_mps2,
        horizon_s,
        radius_m,
    )
    if not all(isfinite(float(value)) for value in values):
        return False
    if horizon_s < 0.0 or radius_m <= 0.0:
        return False
    if ego_station_m > 0.0:
        # Once ego owns the conflict point, yielding here can deadlock every
        # approach. Cross-traffic remains SUMO-controlled while ego clears it.
        return False

    radius_squared = radius_m**2
    if ego_station_m**2 + object_station_m**2 <= radius_squared:
        return True

    forward_acceleration_mps2 = max(
        0.0,
        previous_command_mps2,
        policy_command_mps2,
    )
    projected_ego_velocity_mps = max(0.0, ego_speed_mps) + (
        0.5 * forward_acceleration_mps2 * horizon_s
    )
    velocity_squared = (
        projected_ego_velocity_mps**2 + object_speed_mps**2
    )
    if velocity_squared <= 1e-12:
        return False

    closest_time_s = -(
        ego_station_m * projected_ego_velocity_mps
        + object_station_m * object_speed_mps
    ) / velocity_squared
    closest_time_s = max(0.0, min(horizon_s, closest_time_s))
    projected_ego_station_m = (
        ego_station_m + projected_ego_velocity_mps * closest_time_s
    )
    projected_object_station_m = (
        object_station_m + object_speed_mps * closest_time_s
    )
    return (
        projected_ego_station_m**2 + projected_object_station_m**2
        <= radius_squared
    )


def _three_floats(name: str, values: Sequence[float]) -> tuple[float, float, float]:
    """Return the three floats value owned by this longitudinal control optimizer module."""
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return float(values[0]), float(values[1]), float(values[2])


def _objective_cost(
    solution: Sequence[float],
    request: LongitudinalControlRequest,
) -> float:
    """Return the objective cost value owned by this longitudinal control optimizer module."""
    previous = request.previous_command_mps2
    policy = request.policy_command_mps2
    quadratic = sum(
        solution[row] * solution[column]
        for row in range(len(solution))
        for column in range(len(solution))
    )
    linear = sum((previous - policy) * value for value in solution)
    return quadratic + linear + previous**2 + policy**2 - 2 * previous * policy


def _circle_avoidance_constraint(
    *,
    ego_station_m: float,
    object_station_m: float,
    ego_speed_mps: float,
    object_speed_mps: float,
    horizon_s: float,
    previous_command_mps2: float,
    policy_command_mps2: float,
    side: int,
    radius_m: float,
) -> tuple[float, float]:
    """Return the circle avoidance constraint value owned by this longitudinal control optimizer
    module.
    """
    radius_squared = radius_m**2
    station_norm = ego_station_m**2 + object_station_m**2
    if station_norm <= radius_squared * (1.0 + 1e-12):
        return _circle_contact_constraint(
            ego_station_m=ego_station_m,
            object_station_m=object_station_m,
            ego_speed_mps=ego_speed_mps,
            object_speed_mps=object_speed_mps,
            horizon_s=horizon_s,
            previous_command_mps2=previous_command_mps2,
        )
    object_projection = radius_squared * object_station_m
    discriminant = radius_m**4 * object_station_m**2 - (
        ego_station_m**2 + object_station_m**2
    ) * (radius_m**4 - radius_squared * ego_station_m**2)

    projected_object_station = (
        object_projection + side * sqrt(discriminant)
    ) / station_norm
    projected_ego_station = sqrt(radius_squared - projected_object_station**2)
    if (
        (projected_ego_station - ego_station_m) * projected_ego_station
        + (projected_object_station - object_station_m) * projected_object_station
        >= 1e-5
    ):
        projected_ego_station = -projected_ego_station

    boundary = -(1 + CONSTRAINT_ERROR_MARGIN) * (
        ego_station_m * projected_ego_station
        + object_station_m * projected_object_station
    )
    coefficient = -projected_ego_station * (0.5 * horizon_s**2)
    predicted_ego_station = (
        ego_station_m
        + ego_speed_mps * horizon_s
        + 0.5
        * (previous_command_mps2 - policy_command_mps2)
        * horizon_s**2
    )
    predicted_object_station = object_station_m + object_speed_mps * horizon_s
    bound = boundary - (
        -projected_ego_station * predicted_ego_station
        - projected_object_station * predicted_object_station
    )
    if coefficient > 0.0 and horizon_s > 0.0:
        # The tangent/margin can demand reversing once contact is imminent.
        # Keep its strongest forward-feasible command: zero horizon travel.
        rollback_delta = -2.0 * ego_speed_mps / horizon_s - previous_command_mps2
        bound = max(bound, coefficient * rollback_delta)
    return coefficient, bound


def _circle_contact_constraint(
    *,
    ego_station_m: float,
    object_station_m: float,
    ego_speed_mps: float,
    object_speed_mps: float,
    horizon_s: float,
    previous_command_mps2: float,
) -> tuple[float, float]:
    """Prevent further inward travel when an exterior tangent does not exist.

    The current radial normal requires n dot displacement >= 0. This permits
    a stationary hold without demanding reverse motion, and permits escape
    as the other actor clears. At zero ego station use an approaching-axis
    hold normal, since the radial normal cannot constrain ego travel there.
    """

    normal_ego = ego_station_m
    normal_object = object_station_m
    if normal_ego == 0.0:
        normal_ego = -1.0
    coefficient = -normal_ego * 0.5 * horizon_s**2
    bound = (
        normal_ego
        * (ego_speed_mps * horizon_s + 0.5 * previous_command_mps2 * horizon_s**2)
        + normal_object * object_speed_mps * horizon_s
    )
    if coefficient > 0.0 and horizon_s > 0.0:
        rollback_delta = -2.0 * ego_speed_mps / horizon_s - previous_command_mps2
        bound = max(bound, coefficient * rollback_delta)
    return coefficient, bound
