<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## Inspect TensorBoard

Inspect is read-only. Choose the model first, then the training, evaluation, or
comparison source to inspect. Session and checkpoint choices define the initial
focus when available; they do not rewrite recorded events.

Each tree entry includes a compact event-log summary. The details panel shows
the logged training/evaluation or combined source time, chart/tag counts,
scalar and histogram counts, wall-time range, step range, event records, event
files, and event-data size. A partial-scan warning means a safety limit was
reached; values marked with `≥` are then lower bounds. For bounded scans the
time marked with `~` is estimated from the TensorBoard event timestamps and
file envelope. These statistics are cached until the event-file inventory
changes.

**Open** starts the native TensorBoard and Analysis Dashboard viewers together.
Their browser windows can be opened independently, and Settings controls which
ones open automatically. Cancel starts no inspection target.
Inference runs never appear as TensorBoard or comparison-pack sources.
