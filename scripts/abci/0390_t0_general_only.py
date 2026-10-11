#!/usr/bin/env python3
"""T0 general-only supplement: shared account cap and automatic continuation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from experiments.t0_general_only import main
if __name__ == '__main__':
    raise SystemExit(main())
