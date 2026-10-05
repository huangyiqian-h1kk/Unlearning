#!/usr/bin/env python3
"""Run directly from a checkout, or use the installed clinicia-experiment command."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiments.cli import main

if __name__ == "__main__":
    main()
