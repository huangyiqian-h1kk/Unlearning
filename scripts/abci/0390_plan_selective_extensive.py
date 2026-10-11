#!/usr/bin/env python3
"""CPU-only selective membership audit; no PBS submission."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from experiments.selective_design import main
if __name__ == '__main__':
    main()
