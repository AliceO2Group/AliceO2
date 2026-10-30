#!/usr/bin/env python3
# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""Reference implementations, with uproot, of what the native reader produces.

    reference_inspect.py <file>     prints the JSON of o2-aod-native-inspect --json

The functions are also used by check_corpus.py. uproot stands in for ROOT here
until a ROOT-based reference dumper is available.
"""

import json
import re
import sys

import awkward as ak
import numpy as np
import pyarrow as pa
import uproot

DF_REGEX = re.compile(r"/?DF_([0-9]+)(|-.*)$")


def highest_cycles(keys):
    result = {}
    for k in keys:
        if k.fName not in result or k.fCycle > result[k.fName].fCycle:
            result[k.fName] = k
    return list(result.values())


def data_frames(f):
    # DataInputDirector's order, without parentFiles (uproot cannot write TMap).
    numbers = []
    for k in f._keys:
        m = DF_REGEX.fullmatch(k.fName)
        if m and int(m.group(1)) not in numbers:
            numbers.append(int(m.group(1)))
    return [f"DF_{n}" for n in sorted(numbers)]


def leaf_class(leaf):
    return re.fullmatch(r"Model_(TLeaf.)_v\d+", type(leaf).__name__).group(1)


def branch_json(b):
    leaf = b.member("fLeaves")[0]
    count = leaf.member("fLeafCount")
    offsets = b.entry_offsets
    baskets = []
    for i in range(b._num_normal_baskets):
        baskets.append([int(b.member("fBasketSeek")[i]), int(b.member("fBasketBytes")[i]),
                        int(offsets[i]), int(offsets[i + 1] - offsets[i]), False])
    for j, basket in enumerate(b.embedded_baskets):
        i = b._num_normal_baskets + j
        baskets.append([0, 0, int(offsets[i]), int(basket.num_entries), True])
    return {
        "name": b.name,
        "leaf": leaf_class(leaf),
        "len": int(leaf.member("fLen")),
        "isUnsigned": bool(leaf.member("fIsUnsigned")),
        "countBranch": count.member("fName") if count is not None else "",
        "entries": int(b.member("fEntries")),
        "totBytes": int(b.member("fTotBytes")),
        "zipBytes": int(b.member("fZipBytes")),
        "baskets": baskets,
    }


def inspect(path):
    f = uproot.open(path)
    rf = f.file
    result = {
        "header": {
            "version": rf._fVersion, "begin": rf._fBEGIN, "end": rf._fEND, "seekFree": rf._fSeekFree,
            "nbytesFree": rf._fNbytesFree, "nbytesName": rf._fNbytesName, "units": rf._fUnits,
            "compress": rf._fCompress, "seekInfo": rf._fSeekInfo, "nbytesInfo": rf._fNbytesInfo,
        },
        "keys": [{"name": k.fName, "class": k.fClassName, "cycle": k.fCycle, "seek": k.fSeekKey,
                  "nbytes": k.fNbytes, "objlen": k.fObjlen} for k in f._keys],
        "dataFrames": [],
    }
    for df in data_frames(f):
        trees = []
        for k in highest_cycles(f[df]._keys):
            if k.fClassName != "TTree":
                continue
            t = f[f"{df}/{k.fName}"]
            trees.append({"name": t.name, "entries": int(t.member("fEntries")), "totBytes": int(t.member("fTotBytes")),
                          "zipBytes": int(t.member("fZipBytes")), "branches": [branch_json(b) for b in t.branches]})
        result["dataFrames"].append({"name": df, "trees": trees})
    return result


ARROW_TYPES = {
    ("TLeafO", False): pa.bool_(), ("TLeafB", False): pa.int8(), ("TLeafB", True): pa.uint8(),
    ("TLeafS", False): pa.int16(), ("TLeafS", True): pa.uint16(), ("TLeafI", False): pa.int32(),
    ("TLeafI", True): pa.uint32(), ("TLeafL", False): pa.int64(), ("TLeafL", True): pa.uint64(),
    ("TLeafF", False): pa.float32(), ("TLeafD", False): pa.float64(),
}


def arrow_table(tree):
    """The dataset table the TTree plugin produces for @a tree."""
    arrays, fields = [], []
    for b in tree.branches:
        if b.name.endswith("_size"):
            continue
        leaf = b.member("fLeaves")[0]
        t = ARROW_TYPES[(leaf_class(leaf), bool(leaf.member("fIsUnsigned")))]
        n = int(leaf.member("fLen"))
        data = b.array(library="ak")
        if leaf.member("fLeafCount") is not None:
            counts = np.asarray(ak.num(data), dtype=np.int64)
            offsets = np.zeros(len(counts) + 1, np.int32)
            np.cumsum(counts, out=offsets[1:])
            values = pa.array(np.asarray(ak.flatten(data)), type=t)
            arr = pa.ListArray.from_arrays(pa.array(offsets, pa.int32()), values)
            t = pa.list_(t)
        elif n > 1:
            values = pa.array(np.asarray(data).reshape(-1), type=t)
            arr = pa.FixedSizeListArray.from_arrays(values, n)
            t = pa.list_(t, n)
        else:
            arr = pa.array(np.asarray(data), type=t)
        arrays.append(arr)
        fields.append(pa.field(b.name, t))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    json.dump(inspect(sys.argv[1]), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
