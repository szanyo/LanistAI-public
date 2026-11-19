<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Glossary

| Term | Meaning |
|---|---|
|---|---|
| Application facade | A typed entry point for one family of use cases. |
| Model plugin | Model-specific configuration, environment factory, policies, and hooks supplied to a shared runtime core. |
| Framework runtime | The shared Ray or SB3 lifecycle implementation selected through the engine registry. |
| Runtime selection | The validated model, framework, session, and checkpoint candidate used for a run. |
| Run ID | A unique identifier that prevents an old worker or callback from mutating a newer run. |
| Session lineage | Parent/child relationships between training sessions and continuation checkpoints. |
| Checkpoint artifact kind | The explicit publication shape (`DIRECTORY`, `FILE`, or `FILE_BUNDLE`) used instead of inferring an unpublished checkpoint path. |
| Manifest | Versioned YAML metadata describing a model, framework, session, checkpoint, pack, or runtime state. |
| Policy observation | The normalized values consumed by the learned policy. |
| Simulation read quality | An optional `SimulationRead` identifies live or fallback data. A fallback retains its simulator error. |
| Held-out library | A fixed evaluation scenario inventory with seeds and input file digests. |
