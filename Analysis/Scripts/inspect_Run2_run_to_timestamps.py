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
Script inspect the CCDB object with run number to timestamp conversion.
Author: Nicolo' Jacazio on 2020-07-08
"""

from update_ccdb import get_ccdb_obj
from ROOT import TFile


def main(converter_ccdb_path="Analysis/Core/RunToTimestamp", dest="/tmp/", verbose=0):
    """
    Given a path in the CCDB downloads the timestamp converte object and inspects it
    """
    get_ccdb_obj(converter_ccdb_path, -1, dest=dest, verbose=1)
    obj_file = TFile(f"{dest}/{converter_ccdb_path}/snapshot.root", "READ")
    obj_file.ls()
    obj = obj_file.Get("ccdb_object")
    obj.print()


if __name__ == "__main__":
    main()
