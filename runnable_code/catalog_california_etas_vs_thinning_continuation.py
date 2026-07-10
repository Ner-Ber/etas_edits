#!/usr/bin/env python3
"""Backward-compatible CLI wrapper for California ETAS vs thinning continuation.

Prefer ``runnable_code/continuation_compare.py`` (generic name).
"""

from __future__ import annotations

from continuation_compare import main

if __name__ == "__main__":
    raise SystemExit(main())
