#!/usr/bin/env python3
"""Stable ConRep launcher over the blob-preserved historical variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conrep.entrypoints import DEFAULT_VARIANT, VARIANTS, run_variant  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=DEFAULT_VARIANT)
    parser.add_argument("--list", action="store_true", help="list preserved variants without loading model dependencies")
    args, passthrough = parser.parse_known_args(argv)
    if args.list:
        for name in sorted(VARIANTS):
            marker = " (default)" if name == DEFAULT_VARIANT else ""
            print(f"{name}\t{VARIANTS[name]}{marker}")
        return 0
    run_variant(args.variant, passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
