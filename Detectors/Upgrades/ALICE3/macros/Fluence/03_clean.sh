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

# \file 03_clean.sh
# \brief Clean intermediate simulation files while preserving the final score and geometry outputs.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# Portable cleanup step for the ALICE 3 Geant4 fluence study.
#
# Supported:
#   ./03_clean.sh
#   sh 03_clean.sh
#   source 03_clean.sh      # bash/zsh/ksh
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

fluence_03_clean_main() (
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

echo
echo "============================================================"
echo " ALICE3 LOCAL SIMULATION — CLEAN"
echo "============================================================"

if [ ! -s "$SCORE_FILE" ]; then
  echo "ERROR: final score is missing/empty: $SCORE_FILE" >&2
  echo "Cleanup aborted." >&2
  exit 1
fi

if [ ! -s "$GEOMETRY_FILE" ]; then
  echo "ERROR: final geometry is missing/empty: $GEOMETRY_FILE" >&2
  echo "Cleanup aborted." >&2
  exit 2
fi

# Remove every direct child of Simulation_files/ except the two final files.
# The three glob patterns also cover hidden files without relying on GNU find.
for item in "$SIM_DIR"/* "$SIM_DIR"/.[!.]* "$SIM_DIR"/..?*; do
  if [ ! -e "$item" ] && [ ! -L "$item" ]; then
    continue
  fi

  name=${item##*/}
  case "$name" in
    ALICE3.txt|ALICE3_geometry.root)
      ;;
    *)
      rm -rf "$item"
      ;;
  esac
done

# Strict order-independent verification.
count=0
unexpected=0

for item in "$SIM_DIR"/* "$SIM_DIR"/.[!.]* "$SIM_DIR"/..?*; do
  if [ ! -e "$item" ] && [ ! -L "$item" ]; then
    continue
  fi

  name=${item##*/}
  count=$((count + 1))

  case "$name" in
    ALICE3.txt|ALICE3_geometry.root)
      ;;
    *)
      echo "ERROR: unexpected final Simulation_files/ entry: $name" >&2
      unexpected=1
      ;;
  esac
done

if [ "$count" -ne 2 ]; then
  echo "ERROR: expected exactly two files after cleanup; found $count." >&2
  exit 3
fi

if [ "$unexpected" -ne 0 ] || \
   [ ! -s "$SIM_DIR/ALICE3.txt" ] || \
   [ ! -s "$SIM_DIR/ALICE3_geometry.root" ]; then
  echo "ERROR: unexpected final Simulation_files/ content." >&2
  exit 4
fi

echo
echo "Cleanup completed."
echo "Simulation_files/ now contains only:"
echo "  ALICE3.txt"
echo "  ALICE3_geometry.root"
echo
)

fluence_03_clean_main "$@"
