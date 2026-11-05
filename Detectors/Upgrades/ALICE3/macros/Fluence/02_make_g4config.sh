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

# \file 02_make_g4config.sh
# \brief Build the Geant4 configuration by combining the standard O2 and ALICE 3 scoring macros.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# Portable Geant4 configuration builder for the ALICE 3 fluence study.
#
# Supported:
#   ./02_make_g4config.sh
#   sh 02_make_g4config.sh
#   source 02_make_g4config.sh      # bash/zsh/ksh
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

fluence_02_make_g4config_main() (
set -eu

if [ -z "${O2_ROOT:-}" ]; then
  echo "ERROR: O2_ROOT is not set. Load your O2 or O2Physics environment first." >&2
  exit 1
fi

STANDARD=$O2_ROOT/share/Detectors/gconfig/g4config.in
SCORING=$PWD/scoring_g4_alice3.in
OUTPUT=$PWD/g4config_alice3_scoring.in

if [ ! -f "$STANDARD" ]; then
  echo "ERROR: cannot find standard O2 Geant4 macro: $STANDARD" >&2
  exit 1
fi

if [ ! -f "$SCORING" ]; then
  echo "ERROR: cannot find scoring macro: $SCORING" >&2
  exit 1
fi

cat "$STANDARD" "$SCORING" > "$OUTPUT"
echo "Created $OUTPUT"
)

fluence_02_make_g4config_main "$@"
