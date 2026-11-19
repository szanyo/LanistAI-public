<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

## General Navigation

LanistAI is designed around keyboard-friendly modal workflows.

- `tab` / `shift+tab`: Move between widgets
- Arrow keys: Move inside dialogs, trees, and selector groups
- `pageup` / `pagedown` / `home` / `end`: Scroll long views
- `enter`: Confirm the focused primary action where available
- `esc`: Close the current dialog; from Home, request a safe application exit
- `f9` / `h` / `?`: Open Help for the current dialog

## Home Screen

- `f1` / `c`: Configure runtime mode, model, framework, session, and checkpoint
- `f2` / `v`: View model/session context
- `f3` / `e`: Edit model metadata
- `f4` / `l`: Load a model
- `f9` / `h` / `?`: Help
- `f10` / `a` / `.`: About

Use the Control Panel to Start, Pause, Resume, or Stop the selected run. While
TensorBoard inspection is active, Finish Inspection closes it and returns to
the configuration workflow.

## Typical Workflow

1. Load or create a model project.
2. Open Configure and select the mode, framework, session, and checkpoint.
3. For Evaluation, also select the visualization and pacing profile.
4. Confirm the configuration, then press Start in the Control Panel.
5. Use Inspect or a comparison pack to review recorded experiment evidence.

Each dialog has a separate Help topic describing its controls, read-only
boundaries, and mutating actions. Opening Help from a dialog shows only that
topic. This central Help appends the same topic documents below, so the full
interface reference can be browsed without maintaining duplicate explanations.

## Runtime Modes

- **Training** learns a policy in the selected session and writes checkpoints
  and TensorBoard telemetry.
- **Evaluation** is a finite benchmark of a trained policy. It normally requires
  a selected session and checkpoint, writes TensorBoard evidence, and can be
  included in comparison packs.
- **Inference** is direct active-use execution of the network from one explicitly
  selected session checkpoint. It does not start TensorBoard and is not a
  comparison-pack source.
- **Inspect** is not a runtime mode. It opens recorded training, evaluation, or
  comparison TensorBoard packs without changing them.

## Active Runs and Safe Shutdown

Model administration, runtime reconfiguration, and comparison-pack mutation are
locked while a managed run is active. View, Help, and About remain available.
Pause, Resume, Stop, natural completion, and failures all use the same owned
cleanup lifecycle. Stopping training requires confirmation and closes that
session for further training from the main panel; create a new session to train
again. Quitting the application also performs managed process cleanup.
