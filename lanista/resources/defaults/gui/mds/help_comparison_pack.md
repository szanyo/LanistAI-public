<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## Comparison Packs

Comparison packs are stable snapshots of selected training or evaluation
TensorBoard runs. Add at least two compatible source runs, review the selected
set, provide a pack name and optional description, then Create.

Existing packs can have their source set or description updated, be renamed,
or be securely deleted. These operations mutate the pack only; the source runs
remain unchanged. Inspecting a pack belongs to the separate read-only Inspect
dialog.

Source and existing-pack tree labels include compact logged-time and chart/tag
statistics. The selected-pack details aggregate copied event files, source
durations, steps, records, and data size without rewriting the pack manifest.
The `~` and `≥` markers identify estimated time and bounded lower-count values
for large event logs.
