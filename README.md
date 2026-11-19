# LanistAI

This repository contains a reinforcement learning workbench and the Susan
autonomous driving experiment. Susan uses SUMO, PPO, generated training scenarios, and a fixed held-out
scenario library. The application includes Stable-Baselines3 and Ray RLlib
integrations.

## Contents

- `lanista/` contains the application and its resources
- `lanista/rl_model_storage/models/susan_project/` contains the Susan model,
  training configuration and the paper's 100-scenario evaluation library
- `requirements.txt` pins the Python environment used on Linux x86-64
- `PUBLICATION-SOURCE.json` records the historical Git references
- `SNAPSHOT-SHA256.json` records the copied files and their SHA-256 digests

## Installation and documentation

Follow the [installation runbook](docs/runbooks/installation.md) to set up
Python and SUMO online or offline and prepare the writable Susan workspace.
Then open the [project documentation](docs/README.md) for the application
workflow, architecture, configuration and evaluation. The
[reproduction guide](docs/reproduction.md) records the paper inputs and the
scope of the included calculations.

## Verification

Run the included numerical contract tests from the repository root after installation.

```bash
python -m unittest \
  lanista.tests.susan.config.test_ppo \
  lanista.tests.susan.reward.test_contextual_wait_penalties \
  lanista.tests.rl_metrics.test_evaluation_grouped_publication
```

## Rights

The source retains its existing copyright notices. The repository does not
grant an open source license. See `AUTHORS.md`.
