"""Compatibility launcher for project-owned ConRep source moved under ``src/``.

The historical path remains executable and importable.  The source filename is
preserved so archived commands continue to select the same implementation.
"""

from pathlib import Path as _Path


_ROOT = _Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src" / "conrep" / "legacy" / _Path(__file__).name
exec(compile(_SOURCE.read_bytes(), str(_SOURCE), "exec"), globals(), globals())
