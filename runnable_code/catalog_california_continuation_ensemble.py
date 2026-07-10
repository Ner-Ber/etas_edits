#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for single-method continuation ensembles.

Prefer ``runnable_code/continuation_ensemble.py`` (generic name).
"""

from __future__ import annotations

from continuation_ensemble import main

if __name__ == "__main__":
    raise SystemExit(main())
