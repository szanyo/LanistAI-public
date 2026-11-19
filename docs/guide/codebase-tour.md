<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Codebase tour and responsibility map

LanistAI is a **modular monolith**. One local process contains the Textual
interface, application use cases, reinforcement-learning runtimes, simulator
adapters, and filesystem persistence. The modules are separated by typed
contracts so a change can stay inside one responsibility even though the
process is not split into network services.

The central reading rule is:

```text
Textual view -> GUI adapter/workflow -> typed application facade
             -> focused service/domain -> port -> external adapter
```

Calls normally move from left to right. Events, immutable results, and state
snapshots return to the left. Shared engine code never reaches upward into the
GUI, and it never reaches sideways into one model project's implementation.

## Top-level map

| Area | Owns | Must not own | Useful starting points |
|---|---|---|---|
| `lanista/application_bootstrap.py`, `application.py`, `app.py` | Early configuration, process lifecycle, composition access, Textual shell, coordinated shutdown | Model algorithms or screen-specific workflows | `LanistAIApp`, `LanistAIApplication`, `ApplicationShutdownService` |
| `lanista/gui/` | Textual screens/widgets, event adapters, immutable UI workflow state, human notifications | Filesystem transactions, framework lifecycle, simulator calls | `gui/screens/home_screen.py`, `gui/widgets/fragments/control_panel.py`, `gui/services/`, `gui/adapters/` |
| `lanista/domains/automotive/` | Framework-neutral vehicle control plus ZalaZONE route and scenario policy | GUI, model-project, Ray/SB3 implementation, or native SUMO ownership | `control/`, `zala_zone/` |
| `lanista/reinforcengine/rl_model/` | Application services, runtime selection/activation, execution lifecycle, manifests, sessions, checkpoints, comparison packs, storage transactions | Textual types or model-specific reward/observation formulas | `application_services.py`, `execution_coordinator.py`, `runtime/activation_service.py`, `sessions/repository_service.py`, `checkpoints/catalog_reader.py`, `comparison/tensorboard_service.py`, `storage/transaction.py` |
| `lanista/reinforcengine/rl_technologies/` | Shared Ray and SB3 runtime, training, evaluation, configuration, and checkpoint mechanics | GUI policy or a particular model's domain decisions | `runtime_contracts.py`, `ray/`, `sb3/` |
| `lanista/reinforcengine/utils/simulation/` | Narrow simulator ports, native SUMO adapter, common episode/context utilities | Framework callbacks or model-specific reward tuning | `ports.py`, `simulation_controller.py` |
| `lanista/reinforcengine/rl_metrics/` | Stable channel/observation names, snapshot projection, aggregation, deterministic tables, and publication policy | Training ownership or screen rendering | `channel_schema.py`, `snapshot_projection.py`, `aggregation.py`, `metric_table.py`, `tensorboard_surface.py` |
| `lanista/rl_model_storage/models/susan_project/resources/` | Thin model plugin, policies, environment wiring, scenario-specific domain behavior | A copied Ray/SB3 lifecycle or unchecked native imports | Susan’s `ray/`, `sb3/` and `utils/` packages |
| `lanista/tests/` | Selected PPO, reward and grouped-report tests | Valuable runtime artifacts as fixtures | The domain directory matching the source responsibility |
| `docs/` | Project guides, architecture, configuration and experiment runbooks | Copied Python signatures or generated HTML | This guide, architecture index, runbooks, extension guides |

## Framework core versus model plugin

Ray and SB3 each have a shared runtime core. A model project supplies policy,
environment construction, scenario/curriculum decisions, checkpoint hooks, and
metric hooks through the typed plugin contract. It should not copy framework
startup, cleanup, checkpoint resolution, or training-loop mechanics.

For Susan, one simulation step is intentionally split into named stages:

```text
simulator input -> scenario context -> control -> command
                -> observation -> reward -> telemetry
```

The model-local pipeline coordinates the stages. Narrow services own numerical
calculation and state. The native SUMO module remains behind the shared
simulator adapter. The [observation](../extensions/observation.md) and [reward](../extensions/reward.md)
pages identify the calculation owners.

## Source packages versus runtime data

The `resources/` package of a model project is source. Training sessions,
checkpoints, TensorBoard events, CSV exports, and logs below model storage are
runtime data. A directory appearing close to Python files does not make it an
architectural source.
