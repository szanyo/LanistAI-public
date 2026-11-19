<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Configuration and manifest reference

This page identifies the configuration and metadata used to run the experiment. The
[model-tuning reference](model-tuning.md) lists the learning parameters shared
by the two backends.

## Configuration layers

| Layer | Typical owner | Contains | Change rule |
|---|---|---|---|
| Process/application configuration | Application bootstrap and composition root | Logging, application data roots, UI/runtime defaults | Edit while the application is stopped and validate startup afterward |
| Runtime selection state | Runtime-selection persistence service | Last successful model/framework selection and revision context | Manage through the application |
| Model manifest and configuration | Model project root | Model identity, timestamps, version, resource paths and framework preference | Use model administration or the versioned migration path |
| Framework manifest | `resources/ray/` or `resources/sb3/` | Framework UUID and model-plugin module/class | Keep module origin below the selected model resource root |
| Session manifest | One training-session directory | Session state, parent lineage and optional source checkpoint | Use session operations |
| Checkpoint manifest | A checkpoint directory or file sidecar | Artifact type, framework, step/curriculum and restore metadata | Preserve the metadata emitted by the checkpoint lifecycle |

Model, session and framework manifests in this layout use integer
`schema_version: 1`. Evaluation library manifests have their own schema.

## Canonical file names

The current storage layout recognizes these canonical names:

| Artifact | Canonical name |
|---|---|
| Model manifest | `model_manifest.yaml` |
| Model configuration | `model_configuration.yml` |
| Framework model configuration | `model_manifest_configuration.yaml` |
| Session manifest | `session_manifest.yml` |
| Checkpoint manifest | `checkpoint_manifest.yml` |

Ray stores checkpoint metadata inside the checkpoint directory. An SB3
checkpoint can also have a ZIP sidecar named `<file>.zip.checkpoint_manifest.yml`.

## Model and framework identity

A model manifest provides a stable UUID in addition to its display name and
version. Application workflows capture that identity and the current revision
before an asynchronous editor or selector is opened. Renaming a display value
does not authorize applying a stale result to another model.

Each supported framework configuration declares a framework identity plus the
model-plugin implementation module and object. The loader verifies that:

- the requested framework and plugin's fixed framework agree
- the module resolves beneath the selected model's resource root
- the object implements the runtime plugin contract
- construction and initialization preserve the declared contract
- candidate initialization succeeds before active-slot publication.

Do not use a free-form module/class value to load code outside a model project.

## Session lineage

A fresh session has no parent. A fork records its parent session and, when
selected, the source checkpoint used for continuation. Creation and deletion
validate the complete affected lineage and commit the model/session metadata
and directories in one unit of work.

A stopped/closed training session is historical evidence. Continue through an
explicit fork instead of reopening it by hand. Evaluation selects a session
and checkpoint without changing training lineage.

## Checkpoint restore contract

A checkpoint binary is restorable only when its canonical manifest supplies
the required framework artifact kind and restore context. Depending on the
framework and mode, that context includes step/curriculum state and integrity
metadata. Catalog presentation may show optional fields, but the restore path
fails closed on required fields.

Do not rename a binary, copy a sidecar from another checkpoint, or edit the
framework UUID to bypass compatibility checks. Recover from a verified backup
or use the reviewed metadata migration path.
