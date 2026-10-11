#!/usr/bin/env python3
"""Isolated rich SFT and pre-unlearning reference evaluation for Qwen3."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from experiments.qwen3_rich import main

if __name__ == "__main__":
    main()
