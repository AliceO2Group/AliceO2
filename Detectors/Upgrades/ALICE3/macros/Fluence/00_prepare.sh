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

# \file 00_prepare.sh
# \brief Prepare the local ALICE 3 Geant4 fluence-study inputs and RD50 NIEL weights.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# Portable preparation step for the ALICE 3 Geant4 fluence study.
#
# Supported:
#   ./00_prepare.sh
#   sh 00_prepare.sh
#   source 00_prepare.sh      # bash/zsh/ksh
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

fluence_00_prepare_main() (
set -eu

HERE=$(fluence_script_dir) || {
  echo "ERROR: cannot determine the directory containing this script." >&2
  echo "Automatic sourced-file discovery is supported in bash, zsh and ksh93." >&2
  echo "For another shell, either execute the script or set:" >&2
  echo "  FLUENCE_SCRIPT_PATH=/full/path/to/this/script.sh" >&2
  exit 1
}

BASE_DIR=${BASE_DIR:-$HERE}
SIM_DIR=${SIM_DIR:-$BASE_DIR/Simulation_files}

FIELD_MACRO=${FIELD_MACRO:-$BASE_DIR/ALICE3Field.C}
SCORING_MACRO=${SCORING_MACRO:-$BASE_DIR/scoring_g4_alice3.in}
G4CONFIG_BUILDER=${G4CONFIG_BUILDER:-$BASE_DIR/02_make_g4config.sh}

# Permanent RD50 files kept beside the code.
RD50_ROOT=${RD50_ROOT:-$BASE_DIR/rd50_niel.root}
RD50_CSV=${RD50_CSV:-$BASE_DIR/rd50_niel.csv}

# Leave empty if the O2/O2Physics environment is already loaded.
# If supplied, this setup file itself must be compatible with POSIX "sh".
O2_SETUP_SCRIPT=${O2_SETUP_SCRIPT:-}

echo
echo "============================================================"
echo " ALICE3 LOCAL SIMULATION — PREPARE"
echo "============================================================"

if [ -n "$O2_SETUP_SCRIPT" ]; then
  if [ ! -r "$O2_SETUP_SCRIPT" ]; then
    echo "ERROR: cannot read O2 setup script: $O2_SETUP_SCRIPT" >&2
    exit 1
  fi
  . "$O2_SETUP_SCRIPT"
fi

for f in "$FIELD_MACRO" "$SCORING_MACRO" "$G4CONFIG_BUILDER"; do
  if [ ! -r "$f" ]; then
    echo "ERROR: missing mandatory file: $f" >&2
    exit 2
  fi
done

if ! command -v o2-sim-serial-run5 >/dev/null 2>&1; then
  echo "ERROR: o2-sim-serial-run5 is not in PATH." >&2
  echo "Load the O2 or O2Physics environment first." >&2
  exit 3
fi

if [ -z "${O2_ROOT:-}" ]; then
  echo "ERROR: O2_ROOT is not set. Load O2 or O2Physics first." >&2
  exit 4
fi

O2_RD50_ROOT=$O2_ROOT/share/Detectors/gconfig/data/rd50_niel.root

if [ ! -r "$O2_RD50_ROOT" ]; then
  echo "ERROR: cannot find $O2_RD50_ROOT" >&2
  exit 5
fi

echo "Refreshing RD50 ROOT from O2:"
echo "  $O2_RD50_ROOT"
echo "-> $RD50_ROOT"
cp -f "$O2_RD50_ROOT" "$RD50_ROOT"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not available." >&2
  exit 6
fi

RD50_CSV_TMP=$RD50_CSV.tmp
rm -f "$RD50_CSV_TMP"

echo
echo "Regenerating $RD50_CSV from $RD50_ROOT"

python3 - "$RD50_ROOT" "$RD50_CSV_TMP" <<'PY'
import sys
import ROOT

input_file, output_file = sys.argv[1], sys.argv[2]

f = ROOT.TFile.Open(input_file, "READ")
if not f or f.IsZombie():
    raise RuntimeError(f"Cannot open {input_file}")

mandatory = [
    (2112, "neutronDW"),
    (2212, "protonDW"),
    (211,  "pionDW"),
]

optional = [
    (11, "electronDW"),
]

graphs = []

for pdg, name in mandatory:
    g = f.Get(name)
    if not g:
        raise RuntimeError(
            f"Mandatory graph '{name}' not found in {input_file}"
        )
    graphs.append((pdg, name, g))

for pdg, name in optional:
    g = f.Get(name)
    if g:
        graphs.append((pdg, name, g))
    else:
        print(f"WARNING: optional graph '{name}' is absent.")

with open(output_file, "w") as out:
    out.write("# pdg,ekin[MeV],weight\n")

    for pdg, name, g in graphs:
        print(f"Writing {name}: PDG {pdg}, {g.GetN()} points")

        for i in range(g.GetN()):
            x = float(g.GetPointX(i))
            y = float(g.GetPointY(i))
            out.write(f"{pdg},{x:.12e},{y:.12e}\n")

f.Close()
print(f"Wrote {output_file}")
PY

if [ ! -s "$RD50_CSV_TMP" ]; then
  echo "ERROR: RD50 CSV generation failed." >&2
  rm -f "$RD50_CSV_TMP"
  exit 7
fi

mv -f "$RD50_CSV_TMP" "$RD50_CSV"

if [ ! -s "$RD50_ROOT" ] || [ ! -s "$RD50_CSV" ]; then
  echo "ERROR: RD50 preparation failed." >&2
  exit 8
fi

mkdir -p "$SIM_DIR"

echo
echo "Preparation OK."
ls -l "$RD50_ROOT" "$RD50_CSV"
echo
echo "Authentication/token files are not touched or checked."
)

fluence_00_prepare_main "$@"
