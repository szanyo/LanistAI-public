<!-- Copyright (c) 2026 Benedek Szanyó; HUN-REN SZTAKI. All rights reserved. -->

## View Model Context

View is a **read-only overview**. Opening or closing it never changes model,
session, checkpoint, or runtime metadata.

- The upper panels show selected model and session facts.
- The compact trees show where each item sits in its lineage.
- **Model Lineage** and **Session Lineage** open a larger read-only tree.
- **Select Session** and **Select Checkpoint** leave View and open the relevant
  selector workflow; a confirmed selection may change the active context.
- **Edit Details** leaves View and opens the mutating Edit dialog.
- **Close** returns without applying anything.

Use arrow keys inside the paired panels and action grid. `F2` or `esc` closes
View; `F3` opens Edit; `F9` opens this topic.
