<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# First run

Complete the [installation runbook](../runbooks/installation.md) first. Use a
separate Susan data directory so the run can write its own sessions and
checkpoints.

## Launch the application

From the repository root, activate the project environment and run:

```bash
python -m lanista
```

The application opens in a terminal through Textual. It runs locally. Use a
terminal large enough to show the Home screen, Control Panel, and footer.

## Confirm the workspace

1. If the run configuration opens at launch, press Esc to return to Home.
2. Press F4 and confirm Susan as the active project. Confirming a candidate
   from the model picker opens its run configuration.
3. Review the framework and mode in the configuration dialog. Press Esc to
   inspect without starting a new training session.
4. Open Help with F9, or use F2 to view the active model context when it is
   available.

Run the [installation smoke check](../runbooks/installation.md#verify-the-installation)
to test scenario generation and a short CPU PPO update in temporary storage.
Use [Using the application](using-the-application.md) for a full run.
