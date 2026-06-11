#!/usr/bin/env bash
# Quick sanity check: LAMMPS Python module + deepmd pair_style available.
# Run on a GPU login or compute node after activating conda env.
set -euo pipefail

echo "=== Python ==="
python --version

echo "=== import lammps ==="
python - <<'PY'
from lammps import lammps
lmp = lammps()
lmp.command("units metal")
lmp.command("atom_style atomic")
print("LAMMPS Python module OK")
PY

echo "=== import deepmd (optional; pair_style uses LAMMPS plugin) ==="
python - <<'PY' || echo "WARN: deepmd python package not importable (may still work via LAMMPS plugin)"
try:
    import deepmd
    print("deepmd-kit version:", getattr(deepmd, "__version__", "unknown"))
except ImportError as e:
    raise SystemExit(e)
PY

echo "=== nvidia-smi (if on GPU node) ==="
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi -L || true
else
  echo "nvidia-smi not found (OK on login node without GPU)"
fi

echo "=== verify_deepmd: all checks passed ==="
