"""Shared pytest fixtures.

Adds `scripts/` to `sys.path` so tests can import the CLI scripts as modules.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
