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

# \file 02_check.sh
# \brief Validate the final ALICE 3 Geant4 fluence-study outputs.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# Portable validation step for the ALICE 3 Geant4 fluence study.
#
# Supported:
#   ./02_check.sh
#   sh 02_check.sh
#   source 02_check.sh      # bash/zsh/ksh
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

fluence_02_check_main() (
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

SCORE_FILE=${SCORE_FILE:-$SIM_DIR/ALICE3.txt}
GEOMETRY_FILE=${GEOMETRY_FILE:-$SIM_DIR/ALICE3_geometry.root}
WORK_DIR=${WORK_DIR:-$SIM_DIR/work}

echo
echo "============================================================"
echo " ALICE3 LOCAL SIMULATION — CHECK"
echo "============================================================"

if [ ! -s "$SCORE_FILE" ]; then
  echo "ERROR: missing/empty score file: $SCORE_FILE" >&2
  exit 1
fi

if [ ! -s "$GEOMETRY_FILE" ]; then
  echo "ERROR: missing/empty geometry file: $GEOMETRY_FILE" >&2
  exit 2
fi

first_line=$(head -n 1 "$SCORE_FILE")

case "$first_line" in
  "# Number of simulated events:"*) ;;
  *)
    echo "ERROR: event-count header missing from ALICE3.txt." >&2
    exit 3
    ;;
esac

echo
echo "Checking scorer blocks..."

missing=0

for scorer in dose neq; do
  if grep -q "primitive scorer name: $scorer" "$SCORE_FILE"; then
    echo "OK: $scorer"
  else
    echo "MISSING: $scorer" >&2
    missing=1
  fi
done

if [ "$missing" -ne 0 ]; then
  echo "ERROR: scorer validation failed." >&2
  exit 4
fi

if grep -q "mesh name: ALICE3" "$SCORE_FILE"; then
  echo "OK: mesh name ALICE3"
else
  echo "WARNING: ALICE3 mesh name was not found in the score file."
  echo "The scorer may have been produced through the legacy compatibility path."
fi

if [ -s "$WORK_DIR/run.log" ]; then
  echo
  echo "Scanning run.log for obvious fatal failures..."

  if ! grep -Ein \
    'segmentation fault|fatal|killed|std::bad_alloc|out of memory|error:' \
    "$WORK_DIR/run.log"
  then
    echo "No obvious fatal message found."
  fi
fi

echo
echo "Final products:"
ls -l "$SCORE_FILE" "$GEOMETRY_FILE"

echo
echo "CHECK OK."
echo
)

fluence_02_check_main "$@"
