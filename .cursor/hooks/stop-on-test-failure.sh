#!/usr/bin/env bash
# Cursor stop: emit followup_message if the last edit-hook pytest run failed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
exec python3 "${ROOT}/.cursor/hooks/pytest_hooks.py" on-stop
