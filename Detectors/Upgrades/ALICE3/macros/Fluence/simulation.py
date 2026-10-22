#!/usr/bin/env python3

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

# \file simulation.py
# \brief Execute the full preparation and running of geant-based fluence simulation
# \author Nicola Nicassio (nicola.nicassio@cern.ch)
# \author Rocco Liotino (rocco.liotino@cern.ch)

"""
file: simulation.py

brief: Command-line controller for preparing, running, checking, and cleaning the ALICE 3 Geant4 fluence study.

author: Nicola Nicassio

author: Rocco Liotino
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ============================================================
# USER SETTINGS — EDIT ONLY HERE IF NEEDED
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
SIM_DIR = BASE_DIR / "Simulation_files"

PREPARE = BASE_DIR / "00_prepare.sh"
RUN = BASE_DIR / "01_run.sh"
CHECK = BASE_DIR / "02_check.sh"
CLEAN = BASE_DIR / "03_clean.sh"

DEFAULT_EVENTS = 15000
DEFAULT_BINS_R = 640
DEFAULT_BINS_Z = 1000
O2_SETUP_SCRIPT: Path | None = None
SIM_HOME: Path | None = None
# ============================================================


def build_env(
    force: bool,
    bins_r: int,
    bins_z: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env["BASE_DIR"] = str(BASE_DIR)
    env["SIM_DIR"] = str(SIM_DIR)

    # Scoring mesh binning passed to 01_run.sh.
    env["SCORING_BINS_R"] = str(bins_r)
    env["SCORING_BINS_Z"] = str(bins_z)

    if O2_SETUP_SCRIPT is not None:
        env["O2_SETUP_SCRIPT"] = str(O2_SETUP_SCRIPT)

    if SIM_HOME is not None:
        env["SIM_HOME"] = str(SIM_HOME)

    if force:
        env["FORCE"] = "1"

    return env


def call(script: Path, args: list[str], env: dict[str, str]) -> None:
    cmd = ["sh", str(script), *args]
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=BASE_DIR, env=env, check=True)


def run_arguments(events: int, seed: int | None) -> list[str]:
    args = [str(events)]
    if seed is not None:
        args.append(str(seed))
    return args


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Local ALICE3 Geant4 scoring workflow"
    )

    parser.add_argument(
        "command",
        choices=("prepare", "run", "check", "clean", "all"),
    )

    parser.add_argument(
        "--events",
        type=int,
        default=DEFAULT_EVENTS,
        help=f"Number of events (default: {DEFAULT_EVENTS})",
    )

    parser.add_argument(
        "--bins-r",
        type=int,
        default=DEFAULT_BINS_R,
        help=f"Number of radial scoring bins (default: {DEFAULT_BINS_R})",
    )

    parser.add_argument(
        "--bins-z",
        type=int,
        default=DEFAULT_BINS_Z,
        help=f"Number of longitudinal scoring bins (default: {DEFAULT_BINS_Z})",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional fixed seed; if omitted, a time-based seed is generated",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing final ALICE3 outputs",
    )

    args = parser.parse_args()

    if args.events <= 0:
        parser.error("--events must be > 0")

    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be >= 0")

    if args.bins_r <= 0:
        parser.error("--bins-r must be > 0")

    if args.bins_z <= 0:
        parser.error("--bins-z must be > 0")

    env = build_env(
        args.force,
        args.bins_r,
        args.bins_z,
    )

    try:
        if args.command == "prepare":
            call(PREPARE, [], env)

        elif args.command == "run":
            call(RUN, run_arguments(args.events, args.seed), env)

        elif args.command == "check":
            call(CHECK, [], env)

        elif args.command == "clean":
            call(CLEAN, [], env)

        elif args.command == "all":
            call(PREPARE, [], env)
            call(RUN, run_arguments(args.events, args.seed), env)
            call(CHECK, [], env)
            call(CLEAN, [], env)

    except subprocess.CalledProcessError as exc:
        print(
            f"\nERROR: workflow stopped with return code {exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
