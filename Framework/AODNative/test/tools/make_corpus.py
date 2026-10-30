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

"""Writes a corpus of AO2D-like files with uproot, for testing the native reader.

    make_corpus.py <outdir> [--big]

Every file has the layout of a DPL AO2D: one directory per timeframe (DF_<n>),
one flat TTree per table, VLA columns as <col>_size + <col>. The same content is
written with each codec and uncompressed. DF numbers are chosen so that numeric
and lexicographic order differ. --big also writes a file larger than 2 GiB.

uproot cannot write TMap (metaData, parentFiles) nor embedded baskets: those
need files written by ROOT.
"""

import sys
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

CODECS = {
    "zstd": uproot.ZSTD(5),
    "zlib": uproot.ZLIB(4),
    "lzma": uproot.LZMA(6),
    "lz4": uproot.LZ4(4),
    "uncompressed": None,
}

# Basket boundaries: each extend() writes one basket per branch. Odd sizes put
# boolean basket boundaries inside bitmap bytes; one basket is larger than the
# decoder's 64 KiB boolean window.
BASKET_SIZES = [3, 13, 70001, 1, 0, 517]

TRACK_TYPES = {
    "fIndexCollisions": np.int32,
    "fFlags": np.uint32,
    "fX": np.float32,
    "fAlpha": np.float64,
    "fTPCNClsFindable": np.uint8,
    "fTPCNClsShared": np.int8,
    "fITSClusterMap": np.uint16,
    "fTRDPattern": np.int16,
    "fGlobalBC": np.uint64,
    "fTime": np.int64,
    "fIsPVContributor": np.bool_,
    "fCov": np.dtype((np.float32, (5,))),
    "fHasDet": np.dtype((np.bool_, (4,))),
    "fNoise": np.uint8,  # random bytes: their baskets do not compress, so they are stored uncompressed
    "fIndexArrayMcParticles": "var * int32",
    "fWeights": "var * float64",
}


def track_chunk(rng, n, start):
    jagged_counts = rng.integers(0, 5, n)
    return {
        "fIndexCollisions": np.arange(start, start + n, dtype=np.int32) // 7,
        "fFlags": rng.integers(0, 2**32, n, dtype=np.uint32),
        "fX": rng.normal(size=n).astype(np.float32),
        "fAlpha": rng.uniform(-np.pi, np.pi, n),
        "fTPCNClsFindable": rng.integers(0, 256, n, dtype=np.uint8),
        "fTPCNClsShared": rng.integers(-128, 128, n, dtype=np.int8),
        "fITSClusterMap": rng.integers(0, 2**16, n, dtype=np.uint16),
        "fTRDPattern": rng.integers(-(2**15), 2**15, n, dtype=np.int16),
        "fGlobalBC": rng.integers(0, 2**63, n, dtype=np.uint64) * 2 + 1,
        "fTime": rng.integers(-(2**62), 2**62, n, dtype=np.int64),
        "fIsPVContributor": rng.integers(0, 2, n).astype(np.bool_),
        "fCov": rng.normal(size=(n, 5)).astype(np.float32),
        "fHasDet": rng.integers(0, 2, (n, 4)).astype(np.bool_),
        "fNoise": rng.integers(0, 256, n, dtype=np.uint8),
        "fIndexArrayMcParticles": ak.unflatten(rng.integers(0, 10**6, jagged_counts.sum(), dtype=np.int32), jagged_counts),
        "fWeights": ak.unflatten(rng.normal(size=jagged_counts.sum()), jagged_counts),
    }


def write_tree(f, path, types, chunks):
    tree = f.mktree(path, types, counter_name=lambda counted: counted + "_size")
    for chunk in chunks:
        tree.extend(chunk)


def write_file(path, compression, seed):
    rng = np.random.default_rng(seed)
    with uproot.recreate(path, compression=compression) as f:
        for df in [2, 10, 1]:
            chunks, start = [], 0
            for n in BASKET_SIZES:
                chunks.append(track_chunk(rng, n, start))
                start += n
            write_tree(f, f"DF_{df}/O2track_iu", TRACK_TYPES, chunks)
            # A small single-basket table, and an empty one.
            write_tree(f, f"DF_{df}/O2bc_001", {"fRunNumber": np.int32, "fGlobalBC": np.uint64},
                       [{"fRunNumber": np.full(5, 544000 + df, np.int32), "fGlobalBC": np.arange(5, dtype=np.uint64)}])
            write_tree(f, f"DF_{df}/O2empty", {"fA": np.float32}, [{"fA": np.zeros(0, np.float32)}])


def write_big(path):
    # About 2.4 GB of barely compressible doubles, in baskets of 440 MB which
    # span many 16 MiB compression chunks, then a small DF whose keys and
    # baskets sit beyond 2 GiB, so that 64-bit seeks are needed to reach it.
    rng = np.random.default_rng(7)
    with uproot.recreate(path, compression=uproot.ZSTD(1)) as f:
        tree = f.mktree("DF_1/O2filler", {"fValue": np.float64})
        for _ in range(6):
            tree.extend({"fValue": rng.normal(size=55_000_000)})
        write_tree(f, "DF_2/O2track_iu", TRACK_TYPES, [track_chunk(rng, 1000, 0), track_chunk(rng, 10, 1000)])


def write_rntuple(path):
    # uproot writes an RNTuple when assigning a dict: the reader must refuse it cleanly.
    with uproot.recreate(path) as f:
        f["DF_1-O2track"] = {"fX": np.arange(10, dtype=np.float32)}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for i, (name, compression) in enumerate(CODECS.items()):
        write_file(out / f"{name}.root", compression, seed=i)
    write_rntuple(out / "rntuple.root")
    if "--big" in sys.argv[2:]:
        write_big(out / "big.root")
    return 0


if __name__ == "__main__":
    sys.exit(main())
