#!/usr/bin/env bash
# Run DeepMD DPA-2 MD via scripts/lammps_mlip_run.py (LAMMPS Python API).
#
# Usage (interactive on a GPU node, or called from Slurm):
#   source hyak/env.sh                    # after copying env.example.sh -> env.sh
#   bash hyak/run_dpa2.sh configs/dac.yaml
#   bash hyak/run_dpa2.sh configs/dac.yaml smoke   # short test run
#
# Environment overrides (optional):
#   MD_SIM_ROOT, CONFIG_FILE, DATA_FILE, MLIP_MODEL, EQUIL_STEPS, PROD_STEPS, ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MD_SIM_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

CONFIG="${1:?usage: run_dpa2.sh <configs/foo.yaml> [smoke]}"
SMOKE_MODE="${2:-}"

if [[ -f "$SCRIPT_DIR/env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$SCRIPT_DIR/env.sh"
fi

# Load paths from YAML unless already set
eval "$(python3 "$SCRIPT_DIR/config_to_env.py" "$CONFIG" ${SMOKE_MODE:+$SMOKE_MODE})"

echo "=== MD_Sim DPA-2 run ==="
echo "  root:       $ROOT"
echo "  system:     $SYSTEM"
echo "  config:     $CONFIG_FILE"
echo "  data:       $DATA_FILE"
echo "  model:      $MLIP_MODEL"
echo "  equil/prod: $EQUIL_STEPS / $PROD_STEPS steps"
echo "  out traj:   $OUT_TRAJ"
echo "=========================="

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: missing LAMMPS data file: $DATA_FILE" >&2
  echo "Build the system first, e.g.:" >&2
  echo "  python scripts/build_system.py --config $CONFIG_FILE" >&2
  exit 1
fi

if [[ ! -f "$MLIP_MODEL" ]]; then
  echo "ERROR: missing DeepMD model: $MLIP_MODEL" >&2
  echo "Download DPA-2 and place the .pb file at that path (see hyak/README.md)." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_TRAJ")" "$(dirname "$OUT_DATA")"

python scripts/lammps_mlip_run.py \
  --data "$DATA_FILE" \
  --mlip-model "$MLIP_MODEL" \
  --out-traj "$OUT_TRAJ" \
  --out-data "$OUT_DATA" \
  --temp "$TEMP" \
  --dt-ps "$DT_PS" \
  --equil-steps "$EQUIL_STEPS" \
  --prod-steps "$PROD_STEPS" \
  --dump-every "$DUMP_EVERY"

echo "Done. Trajectory: $OUT_TRAJ"
