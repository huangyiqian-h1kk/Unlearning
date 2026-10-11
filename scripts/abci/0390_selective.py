#!/usr/bin/env python3
"""Frozen selective training campaign: up to three account-wide PBS allocations."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from experiments.selective_campaign import main
if __name__=='__main__':
    raise SystemExit(main())
