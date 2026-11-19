<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## Edit Model Details

Edit is the **metadata-changing counterpart** of View. Text remains local to
the dialog until **Save Changes** is pressed.

- Model Name, Version, and Description update model metadata.
- Session Name and Description are editable only when a session is selected.
- Checkpoint Note is editable only when a checkpoint is selected.
- The metadata panel is read-only context; UUID and framework are not edited.
- The lineage previews and lineage buttons are read-only navigation aids.
- **Delete Model** requests secure deletion through the parent workflow.
- **Cancel** or `esc` discards uncommitted field edits.

Names and versions are normalized and validated on Save. A disabled field
means its backing object is absent, not that the dialog is still loading.
