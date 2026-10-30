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

"""Checks the native reader tools against the uproot reference on a corpus.

    check_corpus.py <bindir> <corpusdir> [<workdir>]

For every file: the JSON of o2-aod-native-inspect must equal the reference;
every tree dumped by o2-aod-native-dump must equal the reference Arrow table,
and the IPC bytes must not depend on --threads or --pread. Then files the
reader must refuse cleanly: an RNTuple, a patched class version, truncations
and a non-ROOT file. Exits non-zero on any failure.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc
import uproot

sys.path.insert(0, str(Path(__file__).parent))
import reference_inspect  # noqa: E402

failures = []


def fail(what):
    failures.append(what)
    print("FAIL", what)


def run(cmd):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True)


def first_difference(a, b, path="$"):
    if type(a) is not type(b):
        return f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: missing on one side"
            d = first_difference(a[k], b[k], f"{path}.{k}")
            if d:
                return d
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d:
                return d
    elif a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def check_inspect(bindir, path):
    r = run([bindir / "o2-aod-native-inspect", path, "--json"])
    if r.returncode != 0:
        fail(f"{path.name}: inspect failed: {r.stderr.strip()}")
        return None
    native = json.loads(r.stdout)
    native["header"].pop("nfree")  # uproot does not keep it
    reference = reference_inspect.inspect(path)
    d = first_difference(native, reference)
    if d:
        fail(f"{path.name}: inspect differs from reference at {d}")
    return native


def check_dump(bindir, path, native, work):
    f = uproot.open(path)
    ntrees = 0
    for df in native["dataFrames"]:
        for tree in df["trees"]:
            name = f"{df['name']}/{tree['name']}"
            outputs = {}
            for variant, extra in {"default": [], "threads": ["--threads", "4"], "pread": ["--pread"]}.items():
                out = work / f"{path.stem}-{df['name']}-{tree['name']}-{variant}.arrow"
                r = run([bindir / "o2-aod-native-dump", path, name, out, *extra])
                if r.returncode != 0:
                    fail(f"{path.name} {name} ({variant}): dump failed: {r.stderr.strip()}")
                    break
                outputs[variant] = out.read_bytes()
            else:
                if len(set(outputs.values())) != 1:
                    fail(f"{path.name} {name}: IPC bytes depend on --threads / --pread")
                native_table = pa.ipc.open_stream(outputs["default"]).read_all()
                expected = reference_inspect.arrow_table(f[name])
                if not native_table.schema.equals(expected.schema):
                    fail(f"{path.name} {name}: schema differs\n  native:   {native_table.schema}\n  expected: {expected.schema}")
                elif not native_table.equals(expected):
                    for column in expected.column_names:
                        if not native_table[column].equals(expected[column]):
                            fail(f"{path.name} {name}: column {column} differs")
                            break
                ntrees += 1
    return ntrees


def expect_refusal(bindir, path, label, must_contain):
    r = run([bindir / "o2-aod-native-inspect", path, "--json"])
    if r.returncode != 1:
        fail(f"{label}: expected a clean failure (exit 1), got exit {r.returncode}: {r.stderr.strip()[-300:]}")
    elif must_contain not in r.stderr:
        fail(f"{label}: expected '{must_contain}' in the error, got: {r.stderr.strip()}")


def check_refusals(bindir, corpus, work):
    if (corpus / "rntuple.root").exists():
        expect_refusal(bindir, corpus / "rntuple.root", "RNTuple file", "unsupported layout")
    if not (corpus / "uncompressed.root").exists() or not (corpus / "zstd.root").exists():
        print("skipping the refusal checks: they need uncompressed.root and zstd.root")
        return

    # A TBranch claiming version 12: the reader must refuse it, not misparse it.
    data = bytearray((corpus / "uncompressed.root").read_bytes())
    pos = data.find(b"TBranch\x00")
    assert pos > 0 and data[pos + 12 : pos + 14] == b"\x00\x0d", "TBranch v13 not found where expected"
    data[pos + 13] = 12
    patched = work / "patched-version.root"
    patched.write_bytes(data)
    expect_refusal(bindir, patched, "patched TBranch version", "unsupported version 12 of class TBranch")

    original = (corpus / "zstd.root").read_bytes()
    for fraction in (0.1, 0.5, 0.9, 0.999):
        truncated = work / f"truncated-{fraction}.root"
        truncated.write_bytes(original[: int(len(original) * fraction)])
        expect_refusal(bindir, truncated, f"file truncated to {fraction:.1%}", "")

    garbage = work / "not-root.root"
    garbage.write_bytes(b"this is not a ROOT file" * 10)
    expect_refusal(bindir, garbage, "non-ROOT file", "is not a ROOT file")


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    bindir, corpus = Path(sys.argv[1]), Path(sys.argv[2])
    work = Path(sys.argv[3]) if len(sys.argv) == 4 else Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    for path in sorted(corpus.glob("*.root")):
        if path.name == "rntuple.root":
            continue
        native = check_inspect(bindir, path)
        if native is not None:
            n = check_dump(bindir, path, native, work)
            print(f"ok   {path.name}: inspect matches, {n} trees dumped and compared")
    check_refusals(bindir, corpus, work)
    print(f"\n{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
