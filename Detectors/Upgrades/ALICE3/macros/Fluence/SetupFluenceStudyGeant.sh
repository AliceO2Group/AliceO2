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

# \file SetupFluenceStudyGeant.sh
# \brief Install the ALICE 3 Geant4 fluence-study helper files in the current working directory.
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

# SetupFluenceStudyGeant.sh
#
# ALICE 3 Geant4 fluence-study setup.
#
# The following invocation styles are intentionally supported:
#
#   ./SetupFluenceStudyGeant.sh
#   sh SetupFluenceStudyGeant.sh
#   /some/relative/path/SetupFluenceStudyGeant.sh
#   source /some/path/SetupFluenceStudyGeant.sh        # bash / zsh / ksh93
#
# The source directory is always the directory containing THIS setup file.
# The destination directory is always the current working directory.
#
# The setup file itself is never copied.
#
# The actual work runs inside a subshell, so sourcing this file does not
# change the caller's current directory, shell options, or variables.

fluence_script_path()
{
  # Optional explicit override for shells that do not expose the pathname
  # of a sourced file.
  if [ -n "${FLUENCE_SCRIPT_PATH:-}" ]; then
    printf '%s\n' "$FLUENCE_SCRIPT_PATH"
    return 0
  fi

  # Bash: valid both for execution and sourcing.
  if [ -n "${BASH_VERSION:-}" ]; then
    eval 'printf "%s\n" "${BASH_SOURCE[0]}"'
    return 0
  fi

  # zsh: %x expands to the file currently being executed/sourced.
  if [ -n "${ZSH_VERSION:-}" ]; then
    eval 'printf "%s\n" "${(%):-%x}"'
    return 0
  fi

  # ksh93.
  if [ -n "${KSH_VERSION:-}" ]; then
    _fluence_ksh_path=$(eval 'printf "%s" "${.sh.file}"' 2>/dev/null || true)
    if [ -n "$_fluence_ksh_path" ]; then
      printf '%s\n' "$_fluence_ksh_path"
      return 0
    fi
  fi

  # Generic POSIX shell when EXECUTED:
  # $0 is the path used to launch the script.
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

fluence_setup_main() (
  set -eu

  FORCE=0
  DRY_RUN=0

  usage()
  {
    cat <<'EOF'
Usage:
  ./SetupFluenceStudyGeant.sh [--force] [--dry-run]
  sh SetupFluenceStudyGeant.sh [--force] [--dry-run]
  source SetupFluenceStudyGeant.sh [--force] [--dry-run]

The source files are taken automatically from the directory containing
SetupFluenceStudyGeant.sh.

The destination is always the current working directory.

Options:
  --force    Back up differing existing files and replace them.
  --dry-run  Show what would be done without changing any file.
  -h,--help  Show this help message.

The setup script itself is never copied.
EOF
  }

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --force)
        FORCE=1
        ;;
      --dry-run)
        DRY_RUN=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "ERROR: unknown option: $1" >&2
        echo "Use --help for usage." >&2
        exit 1
        ;;
    esac
    shift
  done

  SCRIPT_PATH=$(fluence_script_path) || {
    echo "ERROR: cannot determine the pathname of SetupFluenceStudyGeant.sh." >&2
    echo >&2
    echo "Execution works with any POSIX /bin/sh." >&2
    echo "Automatic sourcing is supported in bash, zsh and ksh93." >&2
    echo "For another shell, either execute the script or set:" >&2
    echo "  FLUENCE_SCRIPT_PATH=/full/path/to/SetupFluenceStudyGeant.sh" >&2
    exit 2
  }

  SCRIPT_DIR=$(CDPATH= cd "$(dirname "$SCRIPT_PATH")" 2>/dev/null && pwd -P) || {
    echo "ERROR: cannot determine the directory containing:" >&2
    echo "  $SCRIPT_PATH" >&2
    exit 2
  }

  SOURCE_DIR=$SCRIPT_DIR
  DEST_DIR=$(pwd -P)

  # Store the distributed file list in positional parameters.
  # Quoting "$@" makes iteration independent of bash/zsh word-splitting
  # rules and safe for paths containing spaces.
  set -- \
    "00_prepare.sh" \
    "01_run.sh" \
    "02_check.sh" \
    "02_make_g4config.sh" \
    "03_clean.sh" \
    "ALICE3Field.C" \
    "README_LOCAL.txt" \
    "analyze_geant.py" \
    "make_geometry_csv.C" \
    "geometry_points.csv" \
    "scoring_g4_alice3.in" \
    "simulation.py"

  echo
  echo "============================================================"
  echo " ALICE3 GEANT4 FLUENCE STUDY — SETUP"
  echo "============================================================"
  echo "Source      : $SOURCE_DIR"
  echo "Destination : $DEST_DIR"

  if [ "$DRY_RUN" -eq 1 ]; then
    echo "Mode        : DRY RUN"
  elif [ "$FORCE" -eq 1 ]; then
    echo "Mode        : FORCE"
  else
    echo "Mode        : SAFE"
  fi
  echo

  # ----------------------------------------------------------
  # 1. Verify that the source package is complete.
  # ----------------------------------------------------------
  HAS_MISSING=0

  for name in "$@"; do
    if [ ! -f "$SOURCE_DIR/$name" ]; then
      if [ "$HAS_MISSING" -eq 0 ]; then
        echo "ERROR: the Fluence source directory is incomplete:" >&2
        echo "  $SOURCE_DIR" >&2
        echo >&2
        echo "Missing required files:" >&2
      fi
      echo "  $name" >&2
      HAS_MISSING=1
    fi
  done

  if [ "$HAS_MISSING" -ne 0 ]; then
    echo >&2
    echo "No destination files were changed." >&2
    exit 3
  fi

  # ----------------------------------------------------------
  # 2. Classify destination files.
  #
  # Do NOT accumulate filenames in a whitespace-split scalar.
  # Re-scan "$@" instead. This is deliberate: zsh and POSIX sh
  # differ in scalar word-splitting behavior.
  # ----------------------------------------------------------
  HAS_IDENTICAL=0
  HAS_NEW=0
  HAS_DIFFERENT=0

  for name in "$@"; do
    src=$SOURCE_DIR/$name
    dst=$DEST_DIR/$name

    if [ -f "$dst" ] && cmp "$src" "$dst" >/dev/null 2>&1; then
      if [ "$HAS_IDENTICAL" -eq 0 ]; then
        echo "Already up to date:"
      fi
      echo "  [OK]   $name"
      HAS_IDENTICAL=1
    fi
  done

  if [ "$HAS_IDENTICAL" -ne 0 ]; then
    echo
  fi

  for name in "$@"; do
    dst=$DEST_DIR/$name

    if [ ! -e "$dst" ] && [ ! -L "$dst" ]; then
      if [ "$HAS_NEW" -eq 0 ]; then
        echo "New files to copy:"
      fi
      echo "  [NEW]  $name"
      HAS_NEW=1
    fi
  done

  if [ "$HAS_NEW" -ne 0 ]; then
    echo
  fi

  for name in "$@"; do
    src=$SOURCE_DIR/$name
    dst=$DEST_DIR/$name

    if { [ -e "$dst" ] || [ -L "$dst" ]; } &&
       ! { [ -f "$dst" ] && cmp "$src" "$dst" >/dev/null 2>&1; }; then
      if [ "$HAS_DIFFERENT" -eq 0 ]; then
        echo "Existing files that differ from the Fluence versions:"
      fi
      echo "  [DIFF] $name"
      HAS_DIFFERENT=1
    fi
  done

  if [ "$HAS_DIFFERENT" -ne 0 ]; then
    echo
  fi

  # ----------------------------------------------------------
  # 3. Safe default: never silently overwrite.
  # ----------------------------------------------------------
  if [ "$HAS_DIFFERENT" -ne 0 ] && [ "$FORCE" -ne 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      echo "DRY RUN: setup would stop here because differing files exist."
      echo "Use --force to back them up and replace them."
      echo
      exit 0
    fi

    echo "ERROR: setup stopped to protect existing files." >&2
    echo >&2
    echo "No files were changed." >&2
    echo >&2
    echo "To replace the differing files, run again with --force." >&2
    echo "The current differing versions will be backed up first." >&2
    exit 4
  fi

  if [ "$HAS_NEW" -eq 0 ] && [ "$HAS_DIFFERENT" -eq 0 ]; then
    echo "Everything is already up to date."
    echo "No files were changed."
    echo
    exit 0
  fi

  if [ "$DRY_RUN" -eq 1 ]; then
    if [ "$FORCE" -eq 1 ] && [ "$HAS_DIFFERENT" -ne 0 ]; then
      echo "DRY RUN: differing files would first be backed up."
    fi
    echo "DRY RUN: no files were changed."
    echo
    exit 0
  fi

  # ----------------------------------------------------------
  # 4. Back up differing files when --force is used.
  # ----------------------------------------------------------
  BACKUP_DIR=""

  if [ "$FORCE" -eq 1 ] && [ "$HAS_DIFFERENT" -ne 0 ]; then
    timestamp=$(date '+%Y%m%d_%H%M%S')
    BACKUP_DIR=$DEST_DIR/fluence_setup_backup_${timestamp}_$$
    mkdir -p "$BACKUP_DIR"

    echo "Backing up differing destination files to:"
    echo "  $BACKUP_DIR"

    for name in "$@"; do
      src=$SOURCE_DIR/$name
      dst=$DEST_DIR/$name

      if { [ -e "$dst" ] || [ -L "$dst" ]; } &&
         ! { [ -f "$dst" ] && cmp "$src" "$dst" >/dev/null 2>&1; }; then

        if [ -f "$dst" ]; then
          cp -p "$dst" "$BACKUP_DIR/$name"
        else
          echo "ERROR: destination exists but is not a regular file:" >&2
          echo "  $dst" >&2
          echo "Remove or rename it manually and run the setup again." >&2
          exit 5
        fi
      fi
    done

    echo
  fi

  # ----------------------------------------------------------
  # 5. Copy/update the package.
  # ----------------------------------------------------------
  for name in "$@"; do
    src=$SOURCE_DIR/$name
    dst=$DEST_DIR/$name

    if [ ! -e "$dst" ] && [ ! -L "$dst" ]; then
      cp -p "$src" "$dst"
      echo "[COPIED]   $name"
    elif [ -f "$dst" ] && cmp "$src" "$dst" >/dev/null 2>&1; then
      :
    elif [ "$FORCE" -eq 1 ]; then
      cp -p "$src" "$dst"
      echo "[UPDATED]  $name"
    fi
  done

  # Main entry points should be directly executable after copying.
  for name in \
    "00_prepare.sh" \
    "01_run.sh" \
    "02_check.sh" \
    "02_make_g4config.sh" \
    "03_clean.sh" \
    "simulation.py" \
    "analyze_geant.py"
  do
    if [ -f "$DEST_DIR/$name" ]; then
      chmod u+x "$DEST_DIR/$name"
    fi
  done

  echo
  echo "============================================================"
  echo " SETUP COMPLETED"
  echo "============================================================"
  echo "Fluence-study files are now available in:"
  echo "  $DEST_DIR"
  echo
  echo "SetupFluenceStudyGeant.sh was NOT copied."
  echo "It remains in:"
  echo "  $SOURCE_DIR"

  if [ -n "$BACKUP_DIR" ]; then
    echo
    echo "Previous differing files were backed up in:"
    echo "  $BACKUP_DIR"
  fi

  echo
  echo "Typical next step:"
  echo "  python3 simulation.py prepare"
  echo
)

fluence_setup_main "$@"
