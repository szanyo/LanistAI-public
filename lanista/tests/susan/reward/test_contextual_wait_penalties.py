# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Verify observable-context wait costs without privileged supervisor signals."""

import math
import unittest
from dataclasses import fields

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    PEDESTRIAN_AVOIDANCE_RADIUS_M,
    nominal_pedestrian_avoidance_approach_station_m,
    nominal_vehicle_avoidance_approach_station_m,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_T_JUNCTION,
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_RED,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.observation_config import (
    GREEN_LIGHT_IDLE_PENALTY_WEIGHT,
    RED_LIGHT_FAR_WAIT_PENALTY_WEIGHT,
    RED_LIGHT_FAR_WAIT_SLOW_SPEED_SCALE_MPS,
    RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS,
    SIGNAL_STOP_SPEED_TOLERANCE_MPS,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.reward.reward_service import (
    RewardCalculationRequest,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.scenario.intersection_strategy_descriptors import (
    PedestrianCrossingState,
)
from lanista.tests.susan.controller.numerical_service_fixtures import (
    math_service_for,
    numerical_controller,
    reward_service_for,
)

NO_OBJECTS = (-200.0, -200.0, -200.0)
VISIBLE_OBJECT = (12.0, -200.0, -200.0)


def _request(**overrides: object) -> RewardCalculationRequest:
    values: dict[str, object] = {
        "supervised_command_mps2": 0.0,
        "previous_speed_mps": 0.0,
        "current_speed_mps": 0.0,
        "applied_delta_mps2": 0.0,
        "step_counter": 1,
        "actual_acceleration_override_mps2": 0.0,
        "road_speed_limit_mps": 4.0,
        "object_stations_m": NO_OBJECTS,
    }
    values.update(overrides)
    return RewardCalculationRequest(**values)  # type: ignore[arg-type]


class ContextualWaitPenaltyTests(unittest.TestCase):
    """Keep wait handling aligned with policy-visible physical constraints."""

    def test_conservative_experiment_weights_are_explicit(self) -> None:
        self.assertEqual(GREEN_LIGHT_IDLE_PENALTY_WEIGHT, 0.50)
        self.assertEqual(RED_LIGHT_FAR_WAIT_PENALTY_WEIGHT, 0.16)

    def test_episode_progress_does_not_change_reward(self) -> None:
        def reward_at(step_counter: int) -> float:
            controller = numerical_controller()
            return (
                reward_service_for(controller).calculate(_request(step_counter=step_counter)).reward
            )

        self.assertAlmostEqual(reward_at(25), reward_at(90))

    def test_near_projected_object_suppresses_green_idle(self) -> None:
        clear_controller = numerical_controller()
        clear_reward = (
            reward_service_for(clear_controller)
            .calculate(
                _request(
                    traffic_light_state_code=TRAFFIC_LIGHT_STATE_GREEN,
                    upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                )
            )
            .reward
        )

        blocked_controller = numerical_controller()
        blocked_reward = (
            reward_service_for(blocked_controller)
            .calculate(
                _request(
                    traffic_light_state_code=TRAFFIC_LIGHT_STATE_GREEN,
                    upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                    object_stations_m=VISIBLE_OBJECT,
                    ego_station_m=-20.0,
                    policy_object_distances_m=(6.0,),
                    policy_object_speeds_mps=(0.0,),
                    policy_object_kinds=("vehicle",),
                )
            )
            .reward
        )

        self.assertGreater(
            clear_controller.reward_state.speed.diagnostics.green_light_idle_penalty,
            0.0,
        )
        self.assertEqual(
            blocked_controller.reward_state.speed.diagnostics.green_light_idle_penalty,
            0.0,
        )
        self.assertGreater(blocked_reward, clear_reward)

    def test_near_projected_object_suppresses_red_far_wait(self) -> None:
        clear_controller = numerical_controller()
        reward_service_for(clear_controller).calculate(
            _request(
                ego_station_m=-20.0,
                distance_to_stopline_m=20.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
            )
        )

        blocked_controller = numerical_controller()
        reward_service_for(blocked_controller).calculate(
            _request(
                ego_station_m=-20.0,
                distance_to_stopline_m=20.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                object_stations_m=VISIBLE_OBJECT,
                policy_object_distances_m=(6.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
            )
        )

        self.assertGreater(
            clear_controller.reward_state.speed.diagnostics.red_light_far_wait_penalty,
            0.0,
        )
        diagnostics = blocked_controller.reward_state.speed.diagnostics
        self.assertEqual(diagnostics.red_light_far_wait_penalty, 0.0)
        self.assertTrue(blocked_controller.reward_state.speed.blocking_profile_active)

    def test_distant_or_unprojected_objects_do_not_exempt_waiting(self) -> None:
        for state in (TRAFFIC_LIGHT_STATE_RED, TRAFFIC_LIGHT_STATE_GREEN):
            for distances in ((), (60.0,)):
                with self.subTest(state=state, distances=distances):
                    controller = numerical_controller()
                    reward_service_for(controller).calculate(
                        _request(
                            ego_station_m=-20.0,
                            distance_to_stopline_m=20.0,
                            traffic_light_state_code=state,
                            upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                            object_stations_m=VISIBLE_OBJECT,
                            policy_object_distances_m=distances,
                            policy_object_speeds_mps=(0.0,),
                            policy_object_kinds=("vehicle",),
                            policy_object_geometries=("longitudinal_gap",),
                        )
                    )
                    speed = controller.reward_state.speed
                    self.assertFalse(speed.blocking_profile_active)
                    if state == TRAFFIC_LIGHT_STATE_RED:
                        self.assertGreater(speed.diagnostics.red_light_far_wait_penalty, 0.0)
                        self.assertEqual(speed.diagnostics.red_light_distance_to_stopline, 20.0)
                    else:
                        self.assertGreater(speed.diagnostics.green_light_idle_penalty, 0.0)

    def test_far_wait_starts_after_the_shared_stop_quality_zone(self) -> None:
        controller = numerical_controller()
        math = math_service_for(controller)
        boundary = RED_LIGHT_PREFERRED_SPEED_PLATEAU_WIDTH_METERS

        math._calculate_red_light_speed_reward(0.0, boundary, 4.0)
        boundary_penalty = controller.reward_state.speed.diagnostics.red_light_far_wait_penalty
        math._calculate_red_light_speed_reward(0.0, boundary + 0.5, 4.0)

        self.assertEqual(boundary_penalty, 0.0)
        self.assertGreater(
            controller.reward_state.speed.diagnostics.red_light_far_wait_penalty,
            0.0,
        )

    def test_far_wait_has_nonzero_release_gradient_from_standstill(self) -> None:
        controller = numerical_controller()
        service = math_service_for(controller)
        distance_to_stopline_m = 20.0
        probe_speed_mps = 0.001

        service._calculate_red_light_speed_reward(
            0.0,
            distance_to_stopline_m,
            4.0,
        )
        stopped_penalty = (
            controller.reward_state.speed.diagnostics.red_light_far_wait_penalty
        )
        service._calculate_red_light_speed_reward(
            probe_speed_mps,
            distance_to_stopline_m,
            4.0,
        )
        moving_penalty = (
            controller.reward_state.speed.diagnostics.red_light_far_wait_penalty
        )

        self.assertGreater(stopped_penalty, moving_penalty)
        self.assertAlmostEqual(
            moving_penalty / stopped_penalty,
            math.exp(
                -probe_speed_mps / RED_LIGHT_FAR_WAIT_SLOW_SPEED_SCALE_MPS
            ),
        )

    def test_close_direct_leader_replaces_red_far_wait_with_blocking_profile(
        self,
    ) -> None:
        clear_controller = numerical_controller()
        reward_service_for(clear_controller).calculate(
            _request(
                ego_station_m=-20.0,
                distance_to_stopline_m=20.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
            )
        )

        lead_controller = numerical_controller()
        reward_service_for(lead_controller).calculate(
            _request(
                ego_station_m=-20.0,
                distance_to_stopline_m=20.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                object_stations_m=(-0.001, -200.0, -200.0),
                policy_object_distances_m=(0.001,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                ego_lead_vehicle_present=True,
                ego_lead_distance_m=0.001,
                ego_lead_speed_mps=0.0,
            )
        )

        self.assertGreater(
            clear_controller.reward_state.speed.diagnostics.red_light_far_wait_penalty,
            0.0,
        )
        lead_speed = lead_controller.reward_state.speed
        self.assertTrue(lead_speed.blocking_profile_active)
        self.assertEqual(
            lead_speed.diagnostics.blocking_profile_distance_m,
            0.001,
        )
        self.assertEqual(
            lead_speed.diagnostics.red_light_far_wait_penalty,
            0.0,
        )

    def test_reward_request_has_no_private_supervisor_decision_fields(self) -> None:
        names = {field.name for field in fields(RewardCalculationRequest)}
        forbidden = {
            "supervisor_decision",
            "safe_command_min",
            "safe_command_max",
            "binding_constraint",
            "supervisor_intervened",
            "optimizer_cost",
        }

        self.assertTrue(names.isdisjoint(forbidden))

    def test_reward_request_keeps_leader_telemetry_separate_from_object_slots(
        self,
    ) -> None:
        names = {field.name for field in fields(RewardCalculationRequest)}

        self.assertIn("ego_lead_vehicle_present", names)
        self.assertIn("ego_lead_distance_m", names)
        self.assertIn("ego_lead_speed_mps", names)
        self.assertIn("policy_object_geometries", names)
        self.assertNotIn("telemetry_lead_vehicle_present", names)
        self.assertNotIn("telemetry_lead_distance_m", names)

    def test_longitudinal_slot_preserves_gap_and_speed_without_qp_conversion(
        self,
    ) -> None:
        request = _request(
            object_stations_m=(-3.25, -200.0, -200.0),
            policy_object_distances_m=(3.25,),
            policy_object_speeds_mps=(1.75,),
            policy_object_kinds=("vehicle",),
            policy_object_geometries=("longitudinal_gap",),
        )

        obstacle = request.blocking_obstacle

        self.assertIsNotNone(obstacle)
        assert obstacle is not None
        self.assertEqual(obstacle.distance_m, 3.25)
        self.assertEqual(obstacle.speed_mps, 1.75)
        self.assertEqual(obstacle.kind.value, "vehicle")
        self.assertIsNone(obstacle.closing_speed_mps)
        self.assertFalse(obstacle.distance_is_constraint_clearance)

    def test_longitudinal_slot_speed_controls_relative_closing_reward(self) -> None:
        def speed_reward(leader_speed_mps: float) -> tuple[float, float]:
            controller = numerical_controller()
            reward_service_for(controller).calculate(
                _request(
                    current_speed_mps=2.0,
                    object_stations_m=(-6.0, -200.0, -200.0),
                    policy_object_distances_m=(6.0,),
                    policy_object_speeds_mps=(leader_speed_mps,),
                    policy_object_kinds=("vehicle",),
                    policy_object_geometries=("longitudinal_gap",),
                )
            )
            speed = controller.reward_state.speed
            return (
                speed.effective,
                speed.diagnostics.blocking_profile_closing_speed_mps,
            )

        stopped_reward, stopped_closing_speed = speed_reward(0.0)
        moving_reward, moving_closing_speed = speed_reward(2.0)

        self.assertLess(stopped_reward, moving_reward)
        self.assertEqual(stopped_closing_speed, 2.0)
        self.assertEqual(moving_closing_speed, 0.0)

    def test_nearer_stopline_keeps_priority_over_a_far_direct_leader(self) -> None:
        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                ego_station_m=-5.0,
                distance_to_stopline_m=5.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(10.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
            )
        )

        speed = controller.reward_state.speed
        self.assertFalse(speed.blocking_profile_active)
        self.assertEqual(speed.diagnostics.red_light_distance_to_stopline, 5.0)

    def test_equal_stopline_clearance_keeps_priority_over_direct_leader(self) -> None:
        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                ego_station_m=-3.0,
                distance_to_stopline_m=3.0,
                traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
                upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(10.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                policy_object_constraint_clearances_m=(3.0,),
            )
        )

        speed = controller.reward_state.speed
        self.assertFalse(speed.blocking_profile_active)
        self.assertEqual(speed.diagnostics.red_light_distance_to_stopline, 3.0)

    def test_nearest_observed_constraint_wins_across_object_sources(self) -> None:
        direct_leader = _request(
            object_stations_m=(-0.5, -2.0, -200.0),
            policy_object_distances_m=(0.5, 2.0),
            policy_object_speeds_mps=(1.0, 0.0),
            policy_object_kinds=("vehicle", "pedestrian"),
            policy_object_geometries=(
                "longitudinal_gap",
                "conflict_point",
            ),
        ).blocking_obstacle
        policy_pedestrian = _request(
            object_stations_m=(-0.25, -0.5, -200.0),
            policy_object_distances_m=(0.25, 0.5),
            policy_object_speeds_mps=(0.0, 1.0),
            policy_object_kinds=("pedestrian", "vehicle"),
            policy_object_geometries=(
                "conflict_point",
                "longitudinal_gap",
            ),
        ).blocking_obstacle

        self.assertIsNotNone(direct_leader)
        self.assertIsNotNone(policy_pedestrian)
        assert direct_leader is not None
        assert policy_pedestrian is not None
        self.assertEqual(direct_leader.kind.value, "vehicle")
        self.assertEqual(direct_leader.distance_m, 0.5)
        self.assertEqual(policy_pedestrian.kind.value, "pedestrian")
        self.assertEqual(policy_pedestrian.distance_m, 0.25)

    def test_leader_telemetry_alone_does_not_create_an_obstacle(self) -> None:
        valid = _request(
            ego_lead_vehicle_present=True,
            ego_lead_distance_m=1.0,
            ego_lead_speed_mps=0.5,
        )
        absent = _request(
            ego_lead_vehicle_present=False,
            ego_lead_distance_m=1.0,
        )
        invalid = _request(
            ego_lead_vehicle_present=True,
            ego_lead_distance_m=math.nan,
        )

        self.assertIsNone(valid.blocking_obstacle)
        self.assertIsNone(absent.blocking_obstacle)
        self.assertIsNone(invalid.blocking_obstacle)

    def test_longitudinal_slot_is_not_released_like_intersection_conflict_objects(
        self,
    ) -> None:
        request = _request(
            ego_station_m=-5.99,
            upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
            object_stations_m=(-2.0, -200.0, -200.0),
            policy_object_distances_m=(2.0,),
            policy_object_speeds_mps=(0.5,),
            policy_object_kinds=("vehicle",),
            policy_object_geometries=("longitudinal_gap",),
        )

        obstacle = request.blocking_obstacle

        self.assertIsNotNone(obstacle)
        assert obstacle is not None
        self.assertEqual(obstacle.distance_m, 2.0)
        self.assertEqual(obstacle.speed_mps, 0.5)

    def test_reward_request_has_separate_constraint_coordinate(self) -> None:
        names = {field.name for field in fields(RewardCalculationRequest)}

        self.assertIn("constraint_ego_station_m", names)

    def test_legacy_pedestrian_uses_clearance_to_the_qp_circle(self) -> None:
        occupied = _request(
            ego_station_m=-6.0,
            constraint_ego_station_m=-6.0,
            pedestrian_parameters=(
                0.0,
                0.8,
                PedestrianCrossingState.ON_CROSSING,
            ),
        )
        clear = _request(
            ego_station_m=-6.0,
            pedestrian_parameters=(
                1.0,
                0.8,
                PedestrianCrossingState.CLEAR,
            ),
        )

        self.assertEqual(occupied.blocking_distance_m, 1.0)
        self.assertIsNone(clear.blocking_distance_m)

    def test_projected_pedestrian_is_not_duplicated_by_the_legacy_tuple(self) -> None:
        request = _request(
            ego_station_m=-2.0,
            constraint_ego_station_m=-13.0,
            object_stations_m=(0.0, -200.0, -200.0),
            policy_object_distances_m=(2.0,),
            policy_object_speeds_mps=(1.0,),
            policy_object_kinds=("pedestrian",),
            policy_object_geometries=("conflict_point",),
            policy_object_constraint_clearances_m=(8.0,),
            pedestrian_parameters=(
                0.0,
                1.0,
                PedestrianCrossingState.ON_CROSSING,
            ),
        )

        obstacle = request.blocking_obstacle

        self.assertIsNotNone(obstacle)
        assert obstacle is not None
        self.assertEqual(obstacle.distance_m, 8.0)
        self.assertTrue(obstacle.distance_is_constraint_clearance)
        self.assertEqual(obstacle.kind.value, "pedestrian")

    def test_pedestrian_qp_clearance_does_not_add_vehicle_headway(self) -> None:
        controller = numerical_controller()
        boundary_station_m = nominal_pedestrian_avoidance_approach_station_m(0.0)
        clearance_m = 15.5
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=5.5,
                road_speed_limit_mps=8.3333333333,
                constraint_ego_station_m=boundary_station_m - clearance_m,
                object_stations_m=(0.0, -200.0, -200.0),
                policy_object_distances_m=(20.5,),
                policy_object_speeds_mps=(1.0,),
                policy_object_kinds=("pedestrian",),
                policy_object_geometries=("conflict_point",),
                policy_object_constraint_clearances_m=(clearance_m,),
                pedestrian_parameters=(
                    0.0,
                    1.0,
                    PedestrianCrossingState.ON_CROSSING,
                ),
            )
        )

        diagnostics = controller.reward_state.speed.diagnostics
        self.assertEqual(diagnostics.blocking_profile_distance_m, clearance_m)
        self.assertGreater(diagnostics.blocking_profile_preferred_speed_mps, 1.0)

    def test_policy_obstacle_distance_activates_shared_stop_profile(self) -> None:
        """Use QP-relative vehicle clearance once the boundary is reached."""

        controller = numerical_controller()
        request = _request(
            current_speed_mps=2.0,
            ego_station_m=-12.0,
            object_stations_m=VISIBLE_OBJECT,
            policy_object_distances_m=(6.0,),
        )

        reward_service_for(controller).calculate(request)

        speed = controller.reward_state.speed
        self.assertTrue(speed.blocking_profile_active)
        self.assertEqual(speed.diagnostics.blocking_profile_distance_m, 0.0)
        self.assertLess(speed.effective, 0.0)

    def test_vehicle_reward_distance_is_clearance_to_qp_boundary(self) -> None:
        boundary_station_m = nominal_vehicle_avoidance_approach_station_m(
            VISIBLE_OBJECT[0],
        )
        request = _request(
            current_speed_mps=2.0,
            ego_station_m=boundary_station_m - 30.0,
            constraint_ego_station_m=boundary_station_m - 3.0,
            object_stations_m=VISIBLE_OBJECT,
            policy_object_distances_m=(99.0,),
            policy_object_kinds=("vehicle",),
        )

        obstacle = request.blocking_obstacle
        self.assertIsNotNone(obstacle)
        assert obstacle is not None
        self.assertAlmostEqual(obstacle.distance_m, 3.0)
        self.assertTrue(obstacle.distance_is_constraint_clearance)

    def test_longitudinal_leader_uses_aligned_fixed_qp_clearance(self) -> None:
        request = _request(
            current_speed_mps=2.0,
            object_stations_m=(-10.0, -200.0, -200.0),
            policy_object_distances_m=(10.0,),
            policy_object_speeds_mps=(1.25,),
            policy_object_kinds=("vehicle",),
            policy_object_geometries=("longitudinal_gap",),
            policy_object_constraint_clearances_m=(2.0,),
        )

        obstacle = request.blocking_obstacle

        self.assertIsNotNone(obstacle)
        assert obstacle is not None
        self.assertEqual(obstacle.distance_m, 2.0)
        self.assertEqual(obstacle.speed_mps, 1.25)
        self.assertTrue(obstacle.distance_is_constraint_clearance)

    def test_moving_obstacle_following_keeps_positive_profile_reward(self) -> None:
        """Reward safe moving progress instead of flattening the curve to zero."""

        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=0.8,
                road_speed_limit_mps=4.0,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(2.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                policy_object_constraint_clearances_m=(2.0,),
            )
        )

        speed = controller.reward_state.speed
        self.assertTrue(speed.blocking_profile_active)
        self.assertGreater(speed.diagnostics.blocking_profile_reward, 0.0)
        self.assertGreater(speed.effective, 0.0)

    def test_stationary_obstacle_wait_remains_neutral(self) -> None:
        """Do not accumulate positive reward while holding behind an obstacle."""

        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=0.0,
                road_speed_limit_mps=4.0,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(2.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                policy_object_constraint_clearances_m=(2.0,),
            )
        )

        speed = controller.reward_state.speed
        self.assertTrue(speed.blocking_profile_active)
        self.assertEqual(speed.diagnostics.blocking_profile_reward, 0.0)
        self.assertEqual(speed.effective, 0.0)

        throttle_controller = numerical_controller()
        reward_service_for(throttle_controller).calculate(
            _request(
                current_speed_mps=SIGNAL_STOP_SPEED_TOLERANCE_MPS / 2.0,
                policy_action_mps2=0.5,
                road_speed_limit_mps=4.0,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(4.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                policy_object_constraint_clearances_m=(0.0,),
            )
        )
        self.assertLess(throttle_controller.reward_state.speed.effective, 0.0)

    def test_stationary_obstacle_wait_ignores_numerical_closing_speed(self) -> None:
        """Keep a settled QP hold neutral across the configured stop tolerance."""

        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=SIGNAL_STOP_SPEED_TOLERANCE_MPS / 2.0,
                road_speed_limit_mps=4.0,
                object_stations_m=(-10.0, -200.0, -200.0),
                policy_object_distances_m=(4.0,),
                policy_object_speeds_mps=(0.0,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("longitudinal_gap",),
                policy_object_constraint_clearances_m=(0.0,),
            )
        )

        speed = controller.reward_state.speed
        self.assertTrue(speed.blocking_profile_active)
        self.assertEqual(
            speed.diagnostics.blocking_profile_closing_speed_mps,
            0.0,
        )
        self.assertEqual(speed.diagnostics.blocking_profile_reward, 0.0)
        self.assertEqual(speed.effective, 0.0)

    def test_approaching_cross_traffic_uses_radial_closing_speed(self) -> None:
        """Combine perpendicular actor motion in the QP conflict plane."""

        controller = numerical_controller()
        ego_station_m = -13.159383
        object_station_m = -24.019937
        ego_speed_mps = 4.215230
        object_speed_mps = 8.489890
        expected_closing_speed_mps = -(
            ego_station_m * ego_speed_mps
            + object_station_m * object_speed_mps
        ) / math.hypot(ego_station_m, object_station_m)
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=ego_speed_mps,
                road_speed_limit_mps=8.5,
                ego_station_m=ego_station_m,
                constraint_ego_station_m=ego_station_m,
                object_stations_m=(object_station_m, -200.0, -200.0),
                policy_object_distances_m=(4.0,),
                policy_object_speeds_mps=(object_speed_mps,),
                policy_object_kinds=("vehicle",),
                policy_object_geometries=("conflict_point",),
                policy_object_constraint_clearances_m=(0.13,),
            )
        )

        diagnostics = controller.reward_state.speed.diagnostics
        self.assertEqual(
            diagnostics.blocking_profile_obstacle_speed_mps,
            object_speed_mps,
        )
        self.assertAlmostEqual(
            diagnostics.blocking_profile_closing_speed_mps,
            expected_closing_speed_mps,
        )
        self.assertLess(diagnostics.blocking_profile_reward, 0.0)

    def test_vehicle_stop_profile_ends_at_legacy_release_boundary(self) -> None:
        at_release_boundary = _request(
            ego_station_m=-6.0,
            upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
            object_stations_m=VISIBLE_OBJECT,
            policy_object_distances_m=(6.0,),
            policy_object_speeds_mps=(0.0,),
            policy_object_kinds=("vehicle",),
        )
        after_release = _request(
            ego_station_m=-5.99,
            upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
            object_stations_m=VISIBLE_OBJECT,
            policy_object_distances_m=(6.0,),
            policy_object_speeds_mps=(0.0,),
            policy_object_kinds=("vehicle",),
        )

        self.assertEqual(at_release_boundary.blocking_distance_m, 0.0)
        self.assertIsNone(after_release.blocking_distance_m)

    def test_released_vehicle_returns_normal_accelerating_reward(self) -> None:
        def speed_reward(*, with_vehicle: bool) -> tuple[float, bool]:
            controller = numerical_controller()
            reward_service_for(controller).calculate(
                _request(
                    current_speed_mps=2.0,
                    ego_station_m=-5.99,
                    upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
                    object_stations_m=(
                        VISIBLE_OBJECT if with_vehicle else NO_OBJECTS
                    ),
                    policy_object_distances_m=(6.0,) if with_vehicle else (),
                    policy_object_speeds_mps=(0.0,) if with_vehicle else (),
                    policy_object_kinds=("vehicle",) if with_vehicle else (),
                )
            )
            speed = controller.reward_state.speed
            return speed.effective, speed.blocking_profile_active

        released_reward, released_profile = speed_reward(with_vehicle=True)
        clear_reward, clear_profile = speed_reward(with_vehicle=False)

        self.assertFalse(released_profile)
        self.assertFalse(clear_profile)
        self.assertGreater(released_reward, 0.0)
        self.assertAlmostEqual(released_reward, clear_reward)

    def test_clearing_cross_traffic_reduces_radial_closing_penalty(self) -> None:
        def speed_reward(obstacle_speed_mps: float) -> tuple[float, float]:
            controller = numerical_controller()
            reward_service_for(controller).calculate(
                _request(
                    current_speed_mps=2.0,
                    ego_station_m=-12.0,
                    upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
                    object_stations_m=VISIBLE_OBJECT,
                    policy_object_distances_m=(6.0,),
                    policy_object_speeds_mps=(obstacle_speed_mps,),
                    policy_object_kinds=("vehicle",),
                )
            )
            speed = controller.reward_state.speed
            return (
                speed.effective,
                speed.diagnostics.blocking_profile_closing_speed_mps,
            )

        stopped_reward, stopped_closing_speed = speed_reward(0.0)
        moving_reward, moving_closing_speed = speed_reward(2.0)

        self.assertLess(stopped_reward, moving_reward)
        self.assertAlmostEqual(stopped_closing_speed, math.sqrt(2.0))
        self.assertEqual(moving_closing_speed, 0.0)

    def test_clearing_pedestrian_reduces_radial_closing_speed(self) -> None:
        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=2.0,
                ego_station_m=-5.99,
                upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
                object_stations_m=VISIBLE_OBJECT,
                policy_object_distances_m=(3.0,),
                policy_object_speeds_mps=(1.5,),
                policy_object_kinds=("pedestrian",),
            )
        )

        diagnostics = controller.reward_state.speed.diagnostics
        self.assertTrue(controller.reward_state.speed.blocking_profile_active)
        self.assertEqual(diagnostics.blocking_profile_obstacle_speed_mps, 0.0)
        self.assertEqual(diagnostics.blocking_profile_closing_speed_mps, 0.0)

    def test_approaching_pedestrian_uses_perpendicular_closing_speed(self) -> None:
        controller = numerical_controller()
        ego_station_m = -6.0
        object_station_m = -3.0
        ego_speed_mps = 2.0
        pedestrian_speed_mps = 1.5
        expected_closing_speed_mps = -(
            ego_station_m * ego_speed_mps
            + object_station_m * pedestrian_speed_mps
        ) / math.hypot(ego_station_m, object_station_m)
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=ego_speed_mps,
                ego_station_m=ego_station_m,
                constraint_ego_station_m=ego_station_m,
                upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
                object_stations_m=(object_station_m, -200.0, -200.0),
                policy_object_distances_m=(3.0,),
                policy_object_speeds_mps=(pedestrian_speed_mps,),
                policy_object_kinds=("pedestrian",),
                policy_object_geometries=("conflict_point",),
                policy_object_constraint_clearances_m=(2.0,),
            )
        )

        diagnostics = controller.reward_state.speed.diagnostics
        self.assertEqual(diagnostics.blocking_profile_obstacle_speed_mps, 0.0)
        self.assertAlmostEqual(
            diagnostics.blocking_profile_closing_speed_mps,
            expected_closing_speed_mps,
        )

    def test_stationary_ego_is_neutral_to_perpendicular_actor_motion(self) -> None:
        controller = numerical_controller()
        reward_service_for(controller).calculate(
            _request(
                current_speed_mps=SIGNAL_STOP_SPEED_TOLERANCE_MPS / 2.0,
                ego_station_m=-6.0,
                constraint_ego_station_m=-6.0,
                upcoming_intersection_type_code=INTERSECTION_TYPE_T_JUNCTION,
                object_stations_m=(-3.0, -200.0, -200.0),
                policy_object_distances_m=(3.0,),
                policy_object_speeds_mps=(1.5,),
                policy_object_kinds=("pedestrian",),
                policy_object_geometries=("conflict_point",),
                policy_object_constraint_clearances_m=(2.0,),
            )
        )

        diagnostics = controller.reward_state.speed.diagnostics
        self.assertEqual(diagnostics.blocking_profile_closing_speed_mps, 0.0)
        self.assertEqual(diagnostics.blocking_profile_reward, 0.0)

    def test_pedestrian_stop_profile_aligns_with_qp_safety_boundary(self) -> None:
        def speed_reward(
            current_speed_mps: float,
            distance_m: float,
        ) -> tuple[float, bool]:
            controller = numerical_controller()
            reward_service_for(controller).calculate(
                _request(
                    ego_station_m=-distance_m,
                    current_speed_mps=current_speed_mps,
                    road_speed_limit_mps=4.0,
                    pedestrian_parameters=(
                        0.0,
                        1.0,
                        PedestrianCrossingState.ON_CROSSING,
                    ),
                )
            )
            speed = controller.reward_state.speed
            return speed.effective, speed.blocking_profile_active

        stopped_reward, stopped_profile = speed_reward(
            0.0,
            PEDESTRIAN_AVOIDANCE_RADIUS_M,
        )
        creeping_reward, creeping_profile = speed_reward(
            0.01,
            PEDESTRIAN_AVOIDANCE_RADIUS_M,
        )
        inside_qp_reward, inside_qp_profile = speed_reward(
            0.0,
            0.5 * PEDESTRIAN_AVOIDANCE_RADIUS_M,
        )

        self.assertTrue(stopped_profile)
        self.assertTrue(creeping_profile)
        self.assertTrue(inside_qp_profile)
        self.assertEqual(stopped_reward, 0.0)
        self.assertLess(creeping_reward, 0.0)
        self.assertLess(inside_qp_reward, 0.0)

    def test_stopline_distance_is_not_erased_by_objects_or_pedestrians(self) -> None:
        controller = numerical_controller()
        request = _request(
            ego_station_m=-12.0,
            distance_to_stopline_m=12.0,
            object_stations_m=VISIBLE_OBJECT,
            pedestrian_parameters=(-1.0, 0.8, PedestrianCrossingState.ON_CROSSING),
            traffic_light_state_code=TRAFFIC_LIGHT_STATE_RED,
            upcoming_intersection_type_code=INTERSECTION_TYPE_TRAFFIC_LIGHT,
        )
        reward_service_for(controller).calculate(request)
        self.assertTrue(controller.reward_state.speed.blocking_profile_active)
        self.assertEqual(request.distance_to_stopline_m, 12.0)
        self.assertEqual(math_service_for(controller)._get_distance_to_stopline(-12.0), 12.0)

    def test_recorded_queue_crawl_loses_plateau_and_throttle_credit(self) -> None:
        for current_speed in (0.0, 0.052, 0.078):
            with self.subTest(current_speed=current_speed):
                controller = numerical_controller()
                math = math_service_for(controller)
                base = math._calculate_speed_component(
                    current_speed, blocking_distance_m=0.13, road_speed_limit=4.1667,
                    blocking_obstacle_speed_mps=0.0,
                    blocking_distance_is_constraint_clearance=True,
                )
                throttle = math._calculate_speed_component(
                    current_speed, blocking_distance_m=0.13, road_speed_limit=4.1667,
                    blocking_obstacle_speed_mps=0.0,
                    blocking_distance_is_constraint_clearance=True,
                    policy_action_mps2=0.39,
                )
                self.assertLess(base, 0.2)
                self.assertLess(throttle, 0.0)
                self.assertAlmostEqual(base - throttle, (1.0 - 0.13 / 0.5) * 0.39 / 0.5)

    def test_moving_leader_does_not_get_stationary_throttle_penalty(self) -> None:
        math = math_service_for(numerical_controller())
        rewards = [
            math._calculate_speed_component(
                2.0, blocking_distance_m=0.13, road_speed_limit=4.1667,
                blocking_obstacle_speed_mps=2.0,
                blocking_distance_is_constraint_clearance=True,
                policy_action_mps2=command,
            )
            for command in (0.0, 0.8)
        ]
        self.assertEqual(rewards[0], rewards[1])
        self.assertGreater(rewards[1], 0.9)


if __name__ == "__main__":
    unittest.main()
