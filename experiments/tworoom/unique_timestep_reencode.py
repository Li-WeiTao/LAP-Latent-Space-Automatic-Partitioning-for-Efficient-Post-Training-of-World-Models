#!/usr/bin/env python3
"""Compatibility path for the generic backend-driven ``lap-cache`` CLI.

The implementation is intentionally not task-specific. Existing TwoRoom runs
must now supply the LeWM dataset/encoder factories and JSON configurations just
like Push-T or any future backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lap.encoding.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
