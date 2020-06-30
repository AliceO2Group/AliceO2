#!/usr/bin/env python3

# Copyright CERN and copyright holders of ALICE O2. This software is
# distributed under the terms of the GNU General Public License v3 (GPL
# Version 3), copied verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""
Script update the CCDB with run number to timestamp conversion objects.
This is intended to define timestamps for Run2 converted data.
This script works in tandem with `AliRoot/STEER/macros/GetStartAndEndOfRunTimestamp.C`.
The input format is the same as the output of such macro.
Author: Nicolo' Jacazio on 2020-06-30
"""

import argparse
import subprocess


class run_timestamp:
    run = 0
    start = 0
    stop = 0

    def __init__(self, run, start, stop):
        self.run = run
        self.start = start
        self.stop = stop
        self.check()

    def check(self):
        if self.start > self.stop:
            raise ValueError("start > stop", self.start, self.stop)
        if self.start == 0:
            raise ValueError("start is zero")
        if self.stop == 0:
            raise ValueError("stop is zero")

    def __str__(self):
        return f"Run {self.run} start {self.start} stop {self.stop}"

    def __eq__(self, other):
        return self.run == other.run


def main(input_file_name, extra_args, verbose=0):
    """
    Given an input file with line by line runs and start and stop timestamps it updates the dedicated CCDB object
    Extra arguments can be passed to the upload script.
    """
    infile = open(input_file_name)
    run_list = []
    for i in infile:
        i = i.strip()
        i = i.split()
        run = i[1]
        if not run.isdigit():
            raise ValueError("Read run is not a number", run)
        start = i[4]
        if not start.isdigit():
            raise ValueError("Read start is not a number", start)
        stop = i[8]
        if not stop.isdigit():
            raise ValueError("Read stop is not a number", stop)
        entry = run_timestamp(run, start, stop)
        if entry not in run_list:
            run_list.append(entry)
    for i in run_list:
        print("Setting run", i)
        cmd = "o2-analysiscore-makerun2timestamp"
        cmd += f" --run {i.run}"
        cmd += f" --timestamp {i.start}"
        cmd += f" {extra_args}"
        if verbose:
            print(cmd)
        subprocess.run(cmd.split())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Uploads run-number-to-timestamp converter to CCDB from input file.')
    parser.add_argument('input_file_name', metavar='input_file_name', type=str,
                        help='Name of the file with the run and corresponding timestamps')
    parser.add_argument('--extra_args', metavar='Extra_Arguments', type=str,
                        default="",
                        help='Extra arguments for the upload to CCDB')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    args = parser.parse_args()
    main(args.input_file_name, args.extra_args, verbose=args.verbose)
