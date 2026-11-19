<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## Settings

Settings contains application-global preferences, not model metadata. The
startup switch controls whether the LanistAI splash screen is shown when the
application starts. **Maximize terminal** keeps the desktop panel and window
frame when supported. **Fullscreen terminal** is a separate, stronger mode
that hides them. GNOME Terminal on Wayland uses its native window actions;
unsupported terminals ignore either request.
The **Use GPU for training** switch is off by default and applies to SB3
and Ray. When enabled, SB3 uses CUDA and Ray allocates one GPU to its local
learner. When disabled, SB3 uses CPU and Ray allocates zero GPUs. Changing
this setting opens a confirmation dialog. Confirming
saves the choice and immediately restarts LanistAI; cancelling restores the
previous switch position. Stop an active run before changing it.

The two inspection switches control whether the native TensorBoard and
Analysis Dashboard browser pages open automatically after the paired servers
start.

Startup and inspection changes are persisted immediately. The splash setting
applies after restarting LanistAI; inspection settings apply to the next
inspection target.
Turning a switch off does not stop a running server. Use the visible **Close**
button, `F8`, or `esc` to return.

The two manual-open buttons use the inspection lifecycle's current URLs. The
Analysis Dashboard is available for the complete application lifetime at
`http://127.0.0.1:8765/#tensorboard`; native TensorBoard becomes available
after an inspection target is selected. The startup log also exposes the
Analysis URL as a clickable link.
