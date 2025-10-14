#!/usr/bin/env bash
set -euo pipefail

# setup.sh
# Decompress project assets from .zip archives into their expected locations.
# - models.zip → model-server/models/
# - test_files.zip → test-server/test_files/
# - credit_card_transactions.csv.zip → ./credit_card_transactions.csv
#
# Requirements: unzip (zip utilities)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

need_unzip=false
if ! command -v unzip >/dev/null 2>&1; then
  need_unzip=true
fi

if [ "$need_unzip" = true ]; then
  echo "ERROR: 'unzip' is not installed. Please install zip/unzip tools and re-run this script." >&2
  echo "\nInstall hints:" >&2
  echo "- macOS (Homebrew):   brew install zip" >&2
  echo "- Ubuntu/Debian:      sudo apt-get update && sudo apt-get install -y unzip" >&2
  echo "- Fedora:              sudo dnf install -y unzip" >&2
  echo "- CentOS/RHEL (yum):   sudo yum install -y unzip" >&2
  exit 1
fi

# Unzip models.zip into model-server/
if [ -f "models.zip" ]; then
  echo "[setup] Found models.zip. Unzipping into model-server/ ..."
  mkdir -p model-server
  unzip -o models.zip -d model-server >/dev/null
  echo "[setup] Done: model-server/models/ (if contained in the archive)."
else
  echo "[setup] models.zip not found. Skipping."
fi

# Unzip test_files.zip into test-server/
if [ -f "test_files.zip" ]; then
  echo "[setup] Found test_files.zip. Unzipping into test-server/ ..."
  mkdir -p test-server
  unzip -o test_files.zip -d test-server >/dev/null
  echo "[setup] Done: test-server/test_files/ (if contained in the archive)."
else
  echo "[setup] test_files.zip not found. Skipping."
fi

# Unzip credit_card_transactions.csv.zip into project root
if [ -f "credit_card_transactions.csv.zip" ]; then
  echo "[setup] Found credit_card_transactions.csv.zip. Unzipping into repository root ..."
  unzip -o credit_card_transactions.csv.zip -d . >/dev/null
  echo "[setup] Done: ./credit_card_transactions.csv"
else
  echo "[setup] credit_card_transactions.csv.zip not found. Skipping."
fi

echo "[setup] Completed. Verify that the following now exist (if archives were provided):"
echo "- model-server/models/ (contains *.pkl files)"
echo "- test-server/test_files/ (contains X_test.csv / y_test.csv as applicable)"
echo "- ./credit_card_transactions.csv"
