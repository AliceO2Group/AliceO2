#!/bin/sh

# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# \file 01_run.sh
# \brief Run the local ALICE 3 Geant4 TID/NIEL scoring simulation.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# Portable run step for the ALICE 3 Geant4 fluence study.
#
# Supported:
#   ./01_run.sh
#   sh 01_run.sh
#   source 01_run.sh      # bash/zsh/ksh
#
# POSIX dot syntax also works where the shell exposes the sourced filename,
# or when FLUENCE_SCRIPT_PATH is set explicitly.
#
# The main body runs in a subshell so sourcing does not alter the caller's
# shell options or current working directory.

fluence_script_path()
{
  if [ -n "${FLUENCE_SCRIPT_PATH:-}" ]; then
    printf '%s\n' "$FLUENCE_SCRIPT_PATH"
    return 0
  fi

  if [ -n "${BASH_VERSION:-}" ]; then
    eval 'printf "%s\n" "${BASH_SOURCE[0]}"'
    return 0
  fi

  if [ -n "${ZSH_VERSION:-}" ]; then
    eval 'printf "%s\n" "${(%):-%x}"'
    return 0
  fi

  if [ -n "${KSH_VERSION:-}" ]; then
    _fluence_ksh_path=$(eval 'printf "%s" "${.sh.file}"' 2>/dev/null || true)
    if [ -n "$_fluence_ksh_path" ]; then
      printf '%s\n' "$_fluence_ksh_path"
      return 0
    fi
  fi

  case "$0" in
    sh|-sh|*/sh|dash|-dash|*/dash|ash|-ash|*/ash|ksh|-ksh|*/ksh)
      ;;
    *)
      case "$0" in
        */*)
          printf '%s\n' "$0"
          return 0
          ;;
        *)
          _fluence_resolved=$(command -v "$0" 2>/dev/null || true)
          if [ -n "$_fluence_resolved" ] && [ -f "$_fluence_resolved" ]; then
            printf '%s\n' "$_fluence_resolved"
            return 0
          fi
          ;;
      esac
      ;;
  esac

  return 1
}

fluence_script_dir()
{
  _fluence_path=$(fluence_script_path) || return 1
  CDPATH= cd "$(dirname "$_fluence_path")" 2>/dev/null && pwd -P
}

fluence_01_run_main() (
set -eu

HERE=$(fluence_script_dir) || {
  echo "ERROR: cannot determine the directory containing this script." >&2
  echo "Automatic sourced-file discovery is supported in bash, zsh and ksh93." >&2
  echo "For another shell, either execute the script or set:" >&2
  echo "  FLUENCE_SCRIPT_PATH=/full/path/to/this/script.sh" >&2
  exit 1
}

is_uint()
{
  case ${1:-} in
    ''|*[!0-9]*) return 1 ;;
    *) return 0 ;;
  esac
}

utc_timestamp()
{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

BASE_DIR=${BASE_DIR:-$HERE}
SIM_DIR=${SIM_DIR:-$BASE_DIR/Simulation_files}
WORK_DIR=${WORK_DIR:-$SIM_DIR/work}

FIELD_MACRO=${FIELD_MACRO:-$BASE_DIR/ALICE3Field.C}
SCORING_MACRO=${SCORING_MACRO:-$BASE_DIR/scoring_g4_alice3.in}
G4CONFIG_BUILDER=${G4CONFIG_BUILDER:-$BASE_DIR/02_make_g4config.sh}
RD50_CSV=${RD50_CSV:-$BASE_DIR/rd50_niel.csv}

SIM_HOME=${SIM_HOME:-${HOME:-$BASE_DIR}}
O2_SETUP_SCRIPT=${O2_SETUP_SCRIPT:-}

N_EVENTS=${1:-15000}
REQUESTED_SEED=${2:-}

SCORING_BINS_R=${SCORING_BINS_R:-640}
SCORING_BINS_Z=${SCORING_BINS_Z:-1000}

FORCE=${FORCE:-0}

if ! is_uint "$N_EVENTS" || [ "$N_EVENTS" -le 0 ] 2>/dev/null; then
  echo "ERROR: N_EVENTS must be a positive integer." >&2
  exit 1
fi

if ! is_uint "$SCORING_BINS_R" || [ "$SCORING_BINS_R" -le 0 ] 2>/dev/null; then
  echo "ERROR: SCORING_BINS_R must be a positive integer." >&2
  exit 1
fi

if ! is_uint "$SCORING_BINS_Z" || [ "$SCORING_BINS_Z" -le 0 ] 2>/dev/null; then
  echo "ERROR: SCORING_BINS_Z must be a positive integer." >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not available." >&2
  exit 2
fi

if [ -n "$REQUESTED_SEED" ]; then
  if ! is_uint "$REQUESTED_SEED"; then
    echo "ERROR: SEED must be a non-negative integer." >&2
    exit 1
  fi
  SEED=$REQUESTED_SEED
  SEED_SOURCE=user
else
  SEED=$(
    python3 - <<'PY'
import os
import time
print((time.time_ns() ^ (os.getpid() << 16)) % 900_000_000 + 1)
PY
  )
  SEED_SOURCE=time
fi

if [ -n "$O2_SETUP_SCRIPT" ]; then
  if [ ! -r "$O2_SETUP_SCRIPT" ]; then
    echo "ERROR: cannot read O2 setup script: $O2_SETUP_SCRIPT" >&2
    exit 2
  fi
  . "$O2_SETUP_SCRIPT"
fi

if ! command -v o2-sim-serial-run5 >/dev/null 2>&1; then
  echo "ERROR: o2-sim-serial-run5 is not in PATH." >&2
  echo "Load the O2 or O2Physics environment first." >&2
  exit 2
fi

for f in "$FIELD_MACRO" "$SCORING_MACRO" "$G4CONFIG_BUILDER" "$RD50_CSV"; do
  if [ ! -r "$f" ]; then
    echo "ERROR: missing required file: $f" >&2
    echo "Run ./00_prepare.sh first." >&2
    exit 3
  fi
done

FINAL_SCORE=$SIM_DIR/ALICE3.txt
FINAL_GEOMETRY=$SIM_DIR/ALICE3_geometry.root

mkdir -p "$SIM_DIR"

if [ -e "$FINAL_SCORE" ] || [ -e "$FINAL_GEOMETRY" ]; then
  if [ "$FORCE" != "1" ]; then
    echo
    echo "============================================================"
    echo " Previous ALICE3 simulation output found"
    echo "============================================================"
    echo
    echo "The directory:"
    echo "  $SIM_DIR"
    echo
    echo "already contains ALICE3 simulation results."
    echo
    echo "Please rename/move the folder if you want to keep the existing data,"
    echo "or run again with --force to overwrite it."
    echo
    echo "For example:"
    printf '  mv "%s" "%s_old"\n' "$SIM_DIR" "$SIM_DIR"
    echo
    echo "or, using the Python controller:"
    echo "  python3 simulation.py all --events $N_EVENTS --bins-r $SCORING_BINS_R --bins-z $SCORING_BINS_Z --force"
    echo
    exit 4
  fi

  echo
  echo "FORCE enabled: removing previous final ALICE3 outputs."
  rm -f "$FINAL_SCORE" "$FINAL_GEOMETRY"
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

cp -f "$FIELD_MACRO" "$WORK_DIR/ALICE3Field.C"
cp -f "$SCORING_MACRO" "$WORK_DIR/scoring_g4_alice3.in"
cp -f "$G4CONFIG_BUILDER" "$WORK_DIR/02_make_g4config.sh"
cp -f "$RD50_CSV" "$WORK_DIR/rd50_niel.csv"
chmod u+x "$WORK_DIR/02_make_g4config.sh"

# Apply the requested binning ONLY to the work copy.
python3 - \
  "$WORK_DIR/scoring_g4_alice3.in" \
  "$SCORING_BINS_R" \
  "$SCORING_BINS_Z" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
bins_r = int(sys.argv[2])
bins_z = int(sys.argv[3])

text = path.read_text()
pattern = re.compile(
    r"^(\s*/score/mesh/nBin\s+)\d+\s+\d+\s+(\d+\s*)$",
    re.MULTILINE,
)

matches = list(pattern.finditer(text))
if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one active /score/mesh/nBin line in {path}, "
        f"found {len(matches)}"
    )

text = pattern.sub(
    rf"\g<1>{bins_r} {bins_z} \g<2>",
    text,
    count=1,
)
path.write_text(text)
print(f"Scoring mesh bins: nR={bins_r}, nZ={bins_z}, nPhi=1")
PY

export HOME=$SIM_HOME
cd "$WORK_DIR"

echo
echo "============================================================"
echo " ALICE3 LOCAL SIMULATION — RUN"
echo "============================================================"
echo "Events      = $N_EVENTS"
echo "Seed        = $SEED"
echo "Seed source = $SEED_SOURCE"
echo "Bins R      = $SCORING_BINS_R"
echo "Bins Z      = $SCORING_BINS_Z"
echo

./02_make_g4config.sh

G4_CONFIG=$WORK_DIR/g4config_alice3_scoring.in
if [ ! -s "$G4_CONFIG" ]; then
  echo "ERROR: $G4_CONFIG was not created." >&2
  exit 5
fi

export ALICE3_SIM_FIELD=ON
export ALICE3_MAGFIELD_MACRO=$WORK_DIR/ALICE3Field.C

CONFIG_KEYS="MIDBase.mLayout=0; Alice3PassiveBase.mMagnetLayout=MagThickRadius; FT3Base.layoutFT3=kSegmented; IOTOFBase.enableBackwardTOF=false; IOTOFBase.enableForwardTOF=false; G4.g4scoring=true; G4.g4fluenceweight=true; G4.fluenceWeightFile=$WORK_DIR/rd50_niel.csv; G4.configMacroFile=$G4_CONFIG; G4.physicsmode=kFTFP_BERT_HP_optical; SimCutParams.lowneut=true; GlobalSimProcs.CUTNEU=5.e-12"

cat > run_metadata.txt <<EOF
events=$N_EVENTS
seed=$SEED
seed_source=$SEED_SOURCE
bins_r=$SCORING_BINS_R
bins_z=$SCORING_BINS_Z
host=$(uname -n)
start=$(utc_timestamp)
EOF

# POSIX shells do not have a portable "pipefail".  Use a FIFO so that
# the O2 exit status is preserved while tee still displays and saves output.
RUN_FIFO=$WORK_DIR/.o2_run_output_fifo.$$
rm -f "$RUN_FIFO"

if ! mkfifo "$RUN_FIFO"; then
  echo "ERROR: cannot create logging FIFO: $RUN_FIFO" >&2
  exit 5
fi

tee run.log < "$RUN_FIFO" &
TEE_PID=$!

set +e
o2-sim-serial-run5 \
  --detectorList ALICE3 \
  --skipModules ECL \
  --skipModules FCT \
  --skipModules FD3 \
  --skipModules HALL \
  --skipModules MAG \
  -e TGeant4 \
  -n "$N_EVENTS" \
  -g pythia8pp \
  --seed "$SEED" \
  --configKeyValues "$CONFIG_KEYS" \
  > "$RUN_FIFO" 2>&1
O2_STATUS=$?

wait "$TEE_PID"
TEE_STATUS=$?
set -e

rm -f "$RUN_FIFO"
echo "end=$(utc_timestamp)" >> run_metadata.txt

if [ "$O2_STATUS" -ne 0 ]; then
  echo
  echo "============================================================"
  echo " ERROR: o2-sim-serial-run5 failed"
  echo "============================================================"
  echo "Exit code: $O2_STATUS"
  echo "See: $WORK_DIR/run.log"
  echo
  exit "$O2_STATUS"
fi

if [ "$TEE_STATUS" -ne 0 ]; then
  echo "ERROR: tee failed while writing run.log (exit code $TEE_STATUS)." >&2
  exit "$TEE_STATUS"
fi

# Final scorer.
if [ -s "$WORK_DIR/ALICE3.txt" ]; then
  SOURCE_SCORE=$WORK_DIR/ALICE3.txt
else
  score_count=0
  SOURCE_SCORE=""

  for candidate in "$WORK_DIR"/ALICE3.worker*.txt; do
    if [ -s "$candidate" ]; then
      score_count=$((score_count + 1))
      SOURCE_SCORE=$candidate
    fi
  done

  if [ "$score_count" -ne 1 ]; then
    echo "ERROR: expected one ALICE3 scorer, found $score_count." >&2
    exit 6
  fi
fi

first_line=$(head -n 1 "$SOURCE_SCORE" 2>/dev/null || true)
case "$first_line" in
  "# Number of simulated events:"*)
    cp -f "$SOURCE_SCORE" "$FINAL_SCORE"
    ;;
  *)
    {
      echo "# Number of simulated events: $N_EVENTS"
      cat "$SOURCE_SCORE"
    } > "$FINAL_SCORE.tmp"
    mv -f "$FINAL_SCORE.tmp" "$FINAL_SCORE"
    ;;
esac

# Final geometry.
if [ -s "$WORK_DIR/o2sim_geometry.root" ]; then
  SOURCE_GEOMETRY=$WORK_DIR/o2sim_geometry.root
else
  geometry_count=0
  SOURCE_GEOMETRY=""

  for candidate in "$WORK_DIR"/*.root; do
    if [ ! -f "$candidate" ]; then
      continue
    fi

    lower_name=$(basename "$candidate" | tr '[:upper:]' '[:lower:]')
    case "$lower_name" in
      *geometry*.root)
        geometry_count=$((geometry_count + 1))
        SOURCE_GEOMETRY=$candidate
        ;;
    esac
  done

  if [ "$geometry_count" -ne 1 ]; then
    echo "ERROR: expected one geometry ROOT file, found $geometry_count." >&2
    exit 7
  fi
fi

cp -f "$SOURCE_GEOMETRY" "$FINAL_GEOMETRY"

echo
echo "Simulation completed successfully."
ls -l "$FINAL_SCORE" "$FINAL_GEOMETRY"
)

fluence_01_run_main "$@"
