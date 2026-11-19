<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## Configure

Configure stages the runtime mode, model, framework, session, and checkpoint.
Nothing starts until **Confirm** is accepted and **Start** is pressed on Home.

- **Model** opens the model selector; changing it reloads the dependent choices.
- **Mode** chooses Training, Evaluation, or Inference.
- **Framework** selects the implementation available for the loaded model.
- **Session** and **Checkpoint** open their management dialogs.
- Evaluation additionally exposes its visualization and pacing profile.

**New Model** and **New Session** create storage-backed objects. **Inspect** is
read-only. **Compare Pack** opens the pack curation workflow. Disabled controls
mean the current selection or runtime state does not permit that action.

### Evaluation profiles

- **GUI (real-time)** runs the explicitly selected SUMO evaluation scenario
  once and safely restarts the application process after terminal completion.
  The same model, framework, session, checkpoint, and scenario are restored,
  but another run begins only after Start is pressed.
- **Headless (maximum throughput)** removes scheduler pacing delay.
- **Headless (real-time)** retains real-time pacing without the embedded GUI.

Other environment plugins use the same finite evaluation lifecycle without
requiring SUMO.

`enter` confirms, `esc` cancels, and `F9` opens this topic.
