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
        """
        Function to check integrity of timestamp
        """
        if self.start > self.stop:
            raise ValueError("start > stop", self.start, self.stop)
        if self.start == 0:
            raise ValueError("start is zero")
        if self.stop == 0:
            raise ValueError("stop is zero")

        def check_if_int(number, name):
            """
            Function to check if a timestamp is an integer
            """
            if not isinstance(number, int):
                raise ValueError(
                    f"{name} '{number}' is not an integer but a '{type(number)}'")
        check_if_int(self.run, "run")
        check_if_int(self.start, "start")
        check_if_int(self.stop, "stop")

        def check_if_milliseconds(number, name):
            """
            Function to check if a timestamp is in milliseconds
            """
            if not len(str(number)) == 13:
                raise ValueError(f"{name} '{number}' is not in milliseconds")
        check_if_milliseconds(self.start, "start")
        check_if_milliseconds(self.stop, "stop")

    def __str__(self):
        return f"Run {self.run} start {self.start} stop {self.stop}"

    def __eq__(self, other):
        return self.run == other.run


def main(input_file_name, extra_args,
         input_in_seconds=False, delete_previous=True, verbose=False):
    """
    Given an input file with line by line runs and start and stop timestamps it updates the dedicated CCDB object.
    Extra arguments can be passed to the upload script.
    input_in_seconds set to True converts the input from seconds ti milliseconds.
    delete_previous deletes previous uploads in the same path so as to avoid proliferation on CCDB
    verbose flag can be set to 1, 2 to increase the debug level
    URL of ccdb and PATH of objects are passed as default arguments
    """
    infile = open(input_file_name)
    if verbose:
        print(f"Reading run to timestamp from input file '{input_file_name}'")
    run_list = []
    for line_no, i in enumerate(infile):
        i = i.strip()
        if len(i) <= 1:
            continue
        if verbose >= 2:
            print(f"Line number {line_no}: {i}")
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
        if input_in_seconds:
            if verbose:
                print(
                    f"Converting input timestamps start '{start}' and stop '{stop}' from seconds to milliseconds")
            start = f"{start}000"
            stop = f"{stop}000"
        entry = run_timestamp(int(run), int(start), int(stop))
        if entry not in run_list:
            run_list.append(entry)
    print("Will set converter for", len(run_list), "runs")
    successfull = []
    failed = []
    for i in run_list:
        print("Setting run", i)
        cmd = "o2-analysiscore-makerun2timestamp"
        cmd += f" --run {i.run}"
        cmd += f" --timestamp {i.start}"
        if delete_previous:
            cmd += f" --delete_previous 1"
        cmd += f" {extra_args}"
        if i == run_list[-1]:
            # Printing the status of the converter as a last step
            cmd += " -v 1"
        if verbose:
            print(cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = output.decode("utf-8")
        if verbose:
            print(output)
        if "[FATAL] " in output:
            failed.append(i.run)
        else:
            successfull.append(i.run)

    def print_status(counter, msg):
        if len(counter) > 0:
            print(len(counter), msg)
            if verbose >= 3:
                print("Runs:", counter)

    print_status(successfull, "successfully uploaded new runs")
    print_status(
        failed, "failed uploads, retry with option '-vvv' for mor info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Uploads run-number-to-timestamp converter to CCDB from input file.')
    parser.add_argument('input_file_name', metavar='input_file_name', type=str,
                        help='Name of the file with the run and corresponding timestamps')
    parser.add_argument('--extra_args', metavar='Extra_Arguments', type=str,
                        default="",
                        help='Extra arguments for the upload to CCDB. E.g. for the update of the object --extra_args " --update 1"')
    parser.add_argument('--input_in_seconds', '-s', action='count', default=0,
                        help="Use if timestamps taken from input are in seconds")
    parser.add_argument('--delete_previous', '-d', action='count',
                        default=0, help="Deletes previous uploads in the same path so as to avoid proliferation on CCDB")
    parser.add_argument('--verbose', '-v', action='count',
                        default=0, help="Verbose mode 0, 1, 2")
    args = parser.parse_args()
    main(input_file_name=args.input_file_name,
         extra_args=args.extra_args,
         input_in_seconds=args.input_in_seconds,
         verbose=args.verbose)
