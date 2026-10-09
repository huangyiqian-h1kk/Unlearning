#!/usr/bin/env python3
"""Additive ConRep campaign entry. Works from either checkout or frozen snapshot."""
from pathlib import Path
import json
import os
import sys

# Existing campaign controllers can be upgraded without rewriting the frozen
# trainer or changing checkpoint identities. Keep the familiar checkout command.
if __name__ == "__main__" and "--campaign" in sys.argv:
    index = sys.argv.index("--campaign")
    if index + 1 < len(sys.argv):
        plan_path = Path(sys.argv[index + 1]) / "plan.json"
        if plan_path.is_file():
            plan = json.loads(plan_path.read_text())
            controller = plan.get("controller_entry")
            if controller and Path(controller).resolve() != Path(__file__).resolve():
                executable = plan.get("python", sys.executable)
                os.execv(executable, [executable, controller, *sys.argv[1:]])

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from conrep.night.campaign import main

if __name__ == "__main__":
    raise SystemExit(main())
