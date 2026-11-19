<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Troubleshooting and diagnostic decision path

Diagnose from the application boundary inward. Preserve evidence before
changing state, identify the owner and correlation ID, then run the smallest
focused check that can reproduce the failing boundary.

## Capture a minimal evidence bundle

Record the source revision, framework, session, checkpoint, scenario and the
last visible run state. Before restarting, preserve the failure message and
the relevant logs under the [data home](../architecture/storage-and-metadata.md).
Use the operation or run ID to correlate related messages.

## Symptom guide

| Symptom | Check first |
|---|---|
| Application does not reach Home | Use the pinned Python packages and confirm the [installation](../runbooks/installation.md). |
| The model does not appear in the picker | Confirm `LANISTAI_DATA_HOME` is absolute and contains the copied model and its manifest. |
| A checkpoint cannot be restored | Check the selected framework and the [checkpoint metadata](../reference/configuration-and-manifests.md#checkpoint-restore-contract). |
| Stop stays in progress | Check the current worker and cleanup messages before moving or editing the workspace. |
| A simulation operation fails | Keep the scenario ID and the first native cause. Check SUMO, the network and the route inputs. |
| The library is rejected | Verify its manifest and scenario file digests with the [reproduction guide](../reproduction.md#evaluate-the-held-out-cases). |
| A reward or observation differs | Check the scenario, units, command interval and calculation owner. See [reward](../extensions/reward.md) and [observation](../extensions/observation.md). |

## Runtime load

- Confirm whether the failure happened during resolve/origin validation,
  import, construction, initialization, slot commit, or retirement.
- If import/initialization failed, the previous active runtime should remain.
  Its disappearance is a slot-ownership defect.
- If the request was superseded, late progress and candidate publication must
  be ignored. Retire any initialized candidate.
- Do not solve a plugin-origin or contract error by broadening arbitrary import
  paths. Fix the manifest/plugin inside the validated resource root.

## Execution and cleanup

- Use the coordinator snapshot as the state source. A widget label is only a
  projection and may be stale during teardown.
- Match every progress/terminal event to the current `run_id`.
- A missing cleanup event, living worker after a terminal projection, or a
  resource closed twice is a lifecycle defect.
- After a cleanup failure, keep the run evidence and read the first error in the logs.
