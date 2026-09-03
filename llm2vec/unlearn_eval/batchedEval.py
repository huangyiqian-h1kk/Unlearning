"""Compatibility launcher for ClinicIA source moved under ``src/clinicia``.

The legacy path remains executable and importable while callers migrate to the
versioned ClinicIA registry and stable entry point.
"""

from pathlib import Path as _Path


_ROOT = _Path(__file__).resolve().parents[2]
_SOURCE = _ROOT / "src" / "clinicia" / "legacy" / _Path(__file__).name
exec(compile(_SOURCE.read_bytes(), str(_SOURCE), "exec"), globals(), globals())
