<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Using the application

The Home screen is the control surface for runtime selection, model/session
context, execution and inspection. Confirm a selection to apply it through the
application service.

## Navigation

| Key | Home action |
|---|---|
| `F1` or `c` | Configure mode, model, framework, session, and checkpoint |
| `F2` or `v` | View the current model/session context |
| `F3` or `e` | Edit model, session, or checkpoint metadata |
| `F4` or `l` | Load a model runtime |
| `F9`, `h`, or `?` | Open in-application help |
| `F10`, `a`, or `.` | Open About |
| `Esc`, `q`, or `Ctrl+C` | Request coordinated quit |

Inside dialogs, use `Tab`/`Shift+Tab` between controls, arrow keys inside
trees and selector groups, `Enter` for the focused primary action, and `Esc`
to cancel or go back. The footer shows context-sensitive bindings.

## Load and configure a runtime

1. Press F4 and select Susan. Confirm the selection to stage the model and
   reopen the run configuration.
2. Choose Training, Evaluation or Inference, then select SB3 or Ray RLlib.
3. Select the session and checkpoint required by the mode. Confirm the
   configuration to activate it. Use F2 to inspect the active context.

The Configure dialog selects the run. The [PPO settings](../reference/model-tuning.md)
are defined in model-local Python and YAML files.

## Start a training session

1. In Configure, choose **New Session** and **Start Fresh Session** for an
   independent history. To continue a compatible checkpoint, use a fork
   that records the parent session and source checkpoint.
2. Review the session name and confirm the training configuration. A successful
   activation starts training. Watch the Control Panel through `STARTING`
   until it reports `RUNNING`.
3. Use Pause and Resume when the corresponding action is available. To finish
   early, request Stop and keep the application open through `STOPPING`.

Natural completion ends in `COMPLETED`. A successful explicit stop ends in
`STOPPED`. A worker or cleanup error can end the run in `FAILED`.

## Run evaluation

Select Evaluation and an explicit compatible checkpoint. Use the
[held-out evaluation workflow](../runbooks/evaluation-library.md) to run the
paper library. The single-scenario SB3 constant-speed diagnostic is a separate
selection with its own output. Use F2 to verify the active context before
starting evaluation.

## Run inference

1. Open Configure and select **Inference**.
2. Select the framework, session, and an explicit compatible checkpoint.
3. Confirm configuration and use View to verify the complete association.
4. Select **Start**. Pause, Resume, Stop, failure reporting, and cleanup follow
   the same managed execution lifecycle as training and evaluation.

Inference runs the restored policy for active use and does not create a
TensorBoard log directory. It therefore does not appear in Inspect or as a
comparison-pack source. If the checkpoint disappears or no longer belongs to
the selected session and framework, the start fails instead of choosing a
replacement implicitly.

## Inspect TensorBoard data

Inspection is read only from the runtime perspective.

1. Open Configure and choose **Inspect**.
2. In the **Inspect TensorBoard Pack** dialog, select the exact Training,
   Evaluation, or comparison-pack entry. Invalid or missing session manifests
   are reported as invalid entries.
3. TensorBoard is opened for the resolved root-confined log directory.
4. Use **Finish inspection** in the Control Panel when done so the
   application-owned TensorBoard process is closed.

## Safe interruption and quit

Prefer Stop over killing the terminal. Stop waits for the framework, environment,
SUMO and writers to close. If a shutdown step fails, preserve the logs and use
[Troubleshooting](troubleshooting.md).
