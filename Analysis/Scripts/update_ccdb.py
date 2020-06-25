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
TODO add support for user input
"""

import subprocess
from datetime import datetime
import matplotlib.pyplot as plt


def convert_timestamp(ts):
        return datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d %H:%M:%S')


def get_ccdb_obj(path, timestamp, dest = "/tmp/"):
    print("Getting obj", path, "with timestamp",
          timestamp, convert_timestamp(timestamp))
    cmd=f"o2-ccdb-downloadccdbfile --path {path} --dest {dest} --timestamp {timestamp}"
    subprocess.run(cmd.split())


def get_ccdb_obj_validity(path, dest = "/tmp/", verbose = True):
    cmd=f"o2-ccdb-inspectccdbfile {dest}{path}/snapshot.root"
    process=subprocess.Popen(cmd.split(), stdout = subprocess.PIPE)
    output, error=process.communicate()
    output=output.decode("utf-8").split("\n")
    error=error.decode("utf-8").split("\n") if error is not None else error
    if verbose:
        print("out:")
        print(*output, "\n")
        print("err:")
        print(error)
    result=list(filter(lambda x: x.startswith('Valid-'), output))
    ValidFrom = result[0].split()
    ValidUntil = result[1].split()
    print(*output, sep="\n")
    return [int(ValidFrom[-1]), int(ValidUntil[-1])]
    return {ValidFrom[0]: ValidFrom[-1], ValidUntil[0]: ValidUntil[-1]}


def upload_ccdb_obj(path, timestamp_from, timestamp_until, dest="/tmp/", meta=""):
    print("Uploading obj", path, "with timestamp", [timestamp_from, timestamp_until],
          convert_timestamp(timestamp_from), convert_timestamp(timestamp_until))
    key=path.split("/")[-1]
    cmd=f"o2-ccdb-upload -f {dest}{path}/snapshot.root "
    cmd += f"--key {key} --path {path} "
    cmd += f"--starttimestamp {timestamp_from} --endtimestamp {timestamp_until} --meta \"{meta}\""
    subprocess.run(cmd.split())


def main(path, timestamp_from, timestamp_until):
    get_ccdb_obj(path, timestamp_from-1)
    val_before=get_ccdb_obj_validity(path)
    get_ccdb_obj(path, timestamp_until+1)
    val_after=get_ccdb_obj_validity(path)
    overlap_before=val_before[1] > timestamp_from
    overlap_after=val_after[0] < timestamp_until
    print(overlap_before, overlap_after)
    trimmed_before=val_before if not overlap_before else [
        val_before[0], timestamp_from - 1]
    trimmed_after = val_after if not overlap_after else [
        timestamp_until+1, val_after[1]]
    fig, ax = plt.subplots()

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
    main("qc/TOF/TOFTaskCompressed/hDiagnostic",
        1588946517161 + 10000000,
        1588946517161 + 40000000)
