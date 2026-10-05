#!/usr/bin/env python3
"""Manage job-linked logs without importing any training dependency."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.joblogs import main

if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main(ROOT))
