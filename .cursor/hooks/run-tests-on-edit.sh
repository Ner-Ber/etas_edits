#!/usr/bin/env bash
# Cursor afterFileEdit: run mapped pytest for the edited file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "${ROOT}/.cursor/hooks/pytest_hooks.py" on-edit
