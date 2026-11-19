<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Installation runbook

## Prerequisites

Use Linux x86-64 with CPython 3.13 and its virtual environment support.
The tested platform is Ubuntu 24.04.5 LTS with CPython 3.13.18 and glibc 2.39.
Allow at least 15 GB for the Python environment and additional space for
training checkpoints and output. The package list includes CUDA libraries, even when
training on CPU. A working NVIDIA driver is needed only for GPU execution.

## Install the pinned dependencies

From the repository root, create and activate an environment.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Install the pinned packages from a package index.

```bash
python -m pip install -r requirements.txt
python -m pip check
```

## Install offline

The `offline/wheels/` directory allows installation without a package index.

```bash
python tools/verify_snapshot.py --wheels
python -m pip install --no-index --find-links offline/wheels -r requirements.txt
python -m pip check
```

The wheel collection targets Linux x86-64 and CPython 3.13. It occupies about
4.2 GiB and includes files larger than typical Git hosting limits. Git ignores
the wheel directory, so distribute it separately from the source repository.
Copy the complete `wheels` directory into `offline/wheels/`, including its
`SHA256.json` file. Verify it before editing the repository configuration.
The manifest records the digest of each wheel.

On systems where `venv` reports that `ensurepip` is missing, install the
Python 3.13 virtual environment package provided by the operating system. If
the offline wheel collection is present, a new environment can instead be
created without `ensurepip` and bootstrapped from the bundled pip wheel:

```bash
python3.13 tools/verify_snapshot.py --wheels
python3.13 -m venv --without-pip .venv
PYTHONPATH="$PWD/offline/wheels/pip-25.2-py3-none-any.whl" \
  .venv/bin/python -m pip install --no-index --find-links offline/wheels -r requirements.txt
.venv/bin/python -m pip check
source .venv/bin/activate
```

Run these commands from the repository root with no existing `.venv` directory.
They require CPython 3.13 and the matching offline wheel collection.

## Verify the snapshot

Before changing configuration, check the source file digests with
`python tools/verify_snapshot.py`. Add
`--wheels` to check the offline wheels as well and reject unlisted files in the
wheel directory. These checks verify the files listed in the manifests. They
do not authenticate the manifests themselves.
Add `--strict` to report unlisted source files. This ignores `.git`, `.venv`,
`offline`, and Python cache directories.
The repository's `.gitattributes` disables line-ending conversion so Git
preserves the bytes covered by the snapshot and scenario library manifests.

## Configure SUMO

The lockfile pins Eclipse SUMO, libsumo, TraCI and sumolib to 1.27.1.
The Eclipse SUMO package includes the simulator and its route tools. Add
its binaries to the path for the current shell.

```bash
export SUMO_HOME="$(python - <<'PYCODE'
from pathlib import Path
import sumo
print(Path(sumo.__file__).parent)
PYCODE
)"
export PATH="$SUMO_HOME/bin:$PATH"
"$SUMO_HOME/bin/sumo" --version
python -c "import libsumo"
```

Set `sumo_home` in `lanista/resources/defaults/app/config.yaml` to the value
of `SUMO_HOME`. The application reads this configuration at startup. This is a
local change to a file in the snapshot manifest, so a later checksum check
reports that file as modified. Preserve a verified copy when you need the
original snapshot identity.

## Prepare the Susan workspace

Use an absolute directory for writable model and run data. The following
command makes a separate copy of Susan and fails if the target already exists.

```bash
export LANISTAI_DATA_HOME="$HOME/lanistai-data"
python - <<'PY'
import os
import shutil
from pathlib import Path
from lanista.reinforcengine.rl_model.storage.layout import ensure_tensorboard_structure

source = Path('lanista/rl_model_storage/models/susan_project')
target = Path(os.environ['LANISTAI_DATA_HOME']) / 'rl_model_storage/models/susan_project'
target.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, target)
ensure_tensorboard_structure(str(target))
PY
```

Set the same environment variables in each shell used to run the application.
The application writes sessions and measurements under the selected data home.
The repository copy remains the input for creating a new workspace.

## Verify the installation

The following optional check copies the model into a temporary directory,
loads the SB3 plugin, generates 21 training scenarios, and performs an
eight-step CPU PPO update with SUMO.
It uses the model policy with reduced rollout and epoch counts, then saves and
reloads the native SB3 policy and checks its deterministic action. It checks
the local installation without running a complete training session or the
application's session and checkpoint management workflow.

```bash
python tools/smoke_training.py
```

Continue with the [first-run guide](../guide/getting-started.md).
