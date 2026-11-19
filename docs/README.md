<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# LanistAI documentation

LanistAI is a local workbench for designing, running, and comparing
reinforcement-learning experiments across Ray RLlib and Stable-Baselines3.
The included model supplies the driving environment, policy, observations, rewards and SUMO
integration in this source snapshot. The shared engine manages the experiment
lifecycle and calls the SB3 or Ray RLlib runtime.

## Choose a route

| Goal | Start here | Continue with |
|---|---|---|
| Install and launch | [Installation](runbooks/installation.md) | [First run](guide/getting-started.md) |
| Run an experiment | [Using the application](guide/using-the-application.md) | [Reproducing the experiment](reproduction.md) |
| Evaluate a checkpoint on the paper cases | [Held-out evaluation](runbooks/evaluation-library.md) | [Reproduction scope](reproduction.md#scope-of-the-statistical-reports) |
| Browse the paper results | [Paper results and data](results.md) | [Source files](results.md#source-files) |
| Understand the configuration | [Configuration and manifests](reference/configuration-and-manifests.md) | [PPO settings](reference/model-tuning.md) |
| Locate a calculation | [Codebase tour](guide/codebase-tour.md) | [Runtime and simulation](architecture/runtime-and-simulation.md) |
| Trace the policy inputs or reward | [Observation](extensions/observation.md) | [Reward](extensions/reward.md) |
| Locate output or diagnose a problem | [Storage and metadata](architecture/storage-and-metadata.md) | [Troubleshooting](guide/troubleshooting.md) |

The [glossary](glossary.md) defines terms used across these pages.
