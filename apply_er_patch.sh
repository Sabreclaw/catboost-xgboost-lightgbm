#!/usr/bin/env bash

# If the script is invoked without bash (e.g., using sh), re-exec with bash to avoid 'set: Illegal option -o pipefail'
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# Apply the model-server-run example as a patch to the experiment-runner submodule.
# Usage:
#   bash apply_er_patch.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$ROOT_DIR/patches/experiment-runner-model-server-run.patch"
SUBMODULE_DIR="$ROOT_DIR/experiment-runner"

if [[ ! -d "$SUBMODULE_DIR/.git" ]]; then
  echo "ERROR: experiment-runner does not look like a git submodule directory ($SUBMODULE_DIR)." >&2
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: Patch not found: $PATCH_FILE" >&2
  exit 1
fi

echo "Applying patch to experiment-runner ..."
# We apply from inside the submodule so the paths (examples/...) match.
(
  cd "$SUBMODULE_DIR"
  git apply --index "$PATCH_FILE" || git apply "$PATCH_FILE"
)

echo "Patch applied. New files should now exist under:"
echo "  experiment-runner/examples/model-server-run/"
echo
echo "If you want to commit these changes inside the submodule, run:"
echo "  (cd experiment-runner && git add -A && git commit -m 'Add model-server-run example')"
