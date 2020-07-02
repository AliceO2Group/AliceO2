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
Script to update the CCDB with timestamp non-overlapping objects.
If an object is found in the range specified, the object is split into two.
If the requested range was overlapping three objects are uploaded on CCDB:
1) latest object with requested timestamp validity
2) old object with validity [old_lower_validity-requested_lower_bound]
3) old object with validity [requested_upper_bound, old_upper_validity]
Author: Nicolo' Jacazio on 2020-06-22
TODO add support for 3 files update
"""

import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
import argparse


def convert_timestamp(ts):
    """
    Converts the timestamp in milliseconds in human readable format
    """
    return datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')


def get_ccdb_obj(path, timestamp, dest="/tmp/", verbose=0):
    """
    Gets the ccdb object from 'path' and 'timestamp' and downloads it into 'dest'
    """
    if verbose:
        print("Getting obj", path, "with timestamp",
              timestamp, convert_timestamp(timestamp))
    cmd = f"o2-ccdb-downloadccdbfile --path {path} --dest {dest} --timestamp {timestamp}"
    subprocess.run(cmd.split())


def get_ccdb_obj_validity(path, dest="/tmp/", verbose=0):
    """
    Gets the timestamp validity for an object downloaded from CCDB.
    Returns a list with the initial and end timestamps.
    """
    cmd = f"o2-ccdb-inspectccdbfile {dest}{path}/snapshot.root"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    output = output.decode("utf-8").split("\n")
    error = error.decode("utf-8").split("\n") if error is not None else error
    if verbose:
        print("out:")
        print(*output, "\n")
        print("err:")
        print(error)
    result = list(filter(lambda x: x.startswith('Valid-'), output))
    ValidFrom = result[0].split()
    ValidUntil = result[1].split()
    return [int(ValidFrom[-1]), int(ValidUntil[-1])]


def upload_ccdb_obj(path, timestamp_from, timestamp_until, dest="/tmp/", meta=""):
    """
    Uploads a new object to CCDB in the 'path' using the validity timestamp specified
    """
    print("Uploading obj", path, "with timestamp", [timestamp_from, timestamp_until],
          convert_timestamp(timestamp_from), convert_timestamp(timestamp_until))
    key = path.split("/")[-1]
    cmd = f"o2-ccdb-upload -f {dest}{path}/snapshot.root "
    cmd += f"--key {key} --path {path} "
    cmd += f"--starttimestamp {timestamp_from} --endtimestamp {timestamp_until} --meta \"{meta}\""
    subprocess.run(cmd.split())


def main(path, timestamp_from, timestamp_until, verbose=0, show=False):
    """
    Used to upload a new object to CCDB in 'path' valid from 'timestamp_from' to 'timestamp_until'
    Gets the object from CCDB specified in 'path' and for 'timestamp_from-1'
    Gets the object from CCDB specified in 'path' and for 'timestamp_until+1'
    If required plots the situation before and after the update
    """
    get_ccdb_obj(path, timestamp_from-1)
    val_before = get_ccdb_obj_validity(path, verbose=verbose)
    get_ccdb_obj(path, timestamp_until+1)
    val_after = get_ccdb_obj_validity(path, verbose=verbose)
    overlap_before = val_before[1] > timestamp_from
    overlap_after = val_after[0] < timestamp_until
    if verbose:
        if overlap_before:
            print("Previous objects overalps")
        if overlap_after:
            print("Next objects overalps")
    trimmed_before = val_before if not overlap_before else [
        val_before[0], timestamp_from - 1]
    trimmed_after = val_after if not overlap_after else [
        timestamp_until+1, val_after[1]]
    if show:
        fig, ax = plt.subplots()
        fig

        def bef_af(v, y):
            return [v[0] - 1] + v + [v[1] + 1], [0, y, y, 0]
        if True:
            ax.plot(*bef_af(val_before, 0.95), label='before')
            ax.plot(*bef_af(val_after, 1.05), label='after')
        if False:
            ax.plot(*bef_af(trimmed_before, 0.9), label='trimmed before')
            ax.plot(*bef_af(trimmed_after, 1.1), label='trimmed after')
        ax.plot(*bef_af([timestamp_from, timestamp_until], 1), label='object')
        xlim = 10000000
        plt.xlim([timestamp_from-xlim, timestamp_until+xlim])
        plt.ylim(0, 2)
        plt.xlabel('Timestamp')
        plt.ylabel('Validity')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Uploads timestamp non overlapping objects to CCDB."
        "Basic example: `./update_ccdb.py qc/TOF/TOFTaskCompressed/hDiagnostic 1588956517161 1588986517161 --show --verbose`")
    parser.add_argument('path', metavar='path_to_object', type=str,
                        help='Path of the object in the CCDB repository')
    parser.add_argument('timestamp_from', metavar='from_timestamp', type=int,
                        help='Timestamp of start for the new object to use')
    parser.add_argument('timestamp_until', metavar='until_timestamp', type=int,
                        help='Timestamp of stop for the new object to use')
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--show', '-s', action='count', default=0)

    args = parser.parse_args()
    main(path=args.path,
         timestamp_from=args.timestamp_from,
         timestamp_until=args.timestamp_until,
         verbose=args.verbose,
         show=args.show)
