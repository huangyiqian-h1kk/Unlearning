#!/usr/bin/env python3
"""Stable ClinicIA launcher over the preserved evaluation entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clinicia.entrypoints import DEFAULT_ENTRYPOINT, ENTRYPOINTS, run_entrypoint  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrypoint", choices=sorted(ENTRYPOINTS), default=DEFAULT_ENTRYPOINT)
    parser.add_argument("--list", action="store_true", help="list evaluators without loading model dependencies")
    args, passthrough = parser.parse_known_args(argv)
    if args.list:
        for name in sorted(ENTRYPOINTS):
            marker = " (default)" if name == DEFAULT_ENTRYPOINT else ""
            print(f"{name}\t{ENTRYPOINTS[name]}{marker}")
        return 0
    run_entrypoint(args.entrypoint, passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
