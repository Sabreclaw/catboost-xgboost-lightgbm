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

# Validate that experiment-runner looks like a git repo (submodule or standalone clone)
if [[ ! -e "$SUBMODULE_DIR/.git" ]]; then
  # In many submodules, .git is a file (not a dir). As a fallback, probe with git.
  if (cd "$SUBMODULE_DIR" && git rev-parse --is-inside-work-tree >/dev/null 2>&1); then
    : # ok
  else
    echo "ERROR: experiment-runner does not look like a git repository or submodule at: $SUBMODULE_DIR" >&2
    echo "Hint: initialize submodules with:" >&2
    echo "  git submodule update --init --recursive" >&2
    exit 1
  fi
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "ERROR: Patch not found: $PATCH_FILE" >&2
  exit 1
fi

echo "Applying patch to experiment-runner ..."
# We apply from inside the submodule so the paths (examples/...) match.
(
  cd "$SUBMODULE_DIR"

  # If someone mistakenly pasted the patch text into the target files, clean them up first
  BAD_DIR="examples/model-server-run"
  BAD_FILES=("RunnerConfig.py" "README.md")
  mkdir -p "$BAD_DIR"
  for f in "${BAD_FILES[@]}"; do
    if [[ -f "$BAD_DIR/$f" ]]; then
      if grep -q "^diff --git" "$BAD_DIR/$f" 2>/dev/null; then
        ts=$(date +%s)
        echo "Found invalid file with patch text: $BAD_DIR/$f — backing up to $BAD_DIR/$f.bak.$ts and removing"
        mv "$BAD_DIR/$f" "$BAD_DIR/$f.bak.$ts" || true
        git rm -f --cached "$BAD_DIR/$f" >/dev/null 2>&1 || true
        rm -f "$BAD_DIR/$f" || true
      fi
    fi
  done

  # Apply the patch (try with index first for better git integration)
  git apply --index "$PATCH_FILE" || git apply "$PATCH_FILE"
)

echo "Patch applied. New files should now exist under:"
echo "  experiment-runner/examples/model-server-run/"
echo
echo "If you want to commit these changes inside the submodule, run:"
echo "  (cd experiment-runner && git add -A && git commit -m 'Add model-server-run example')"
