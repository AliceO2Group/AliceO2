# ROOT-free AO2D reader: development notes

Work in progress on a DPL plugin which reads Run 3 AO2D files into Arrow
tables without ROOT, falling back to the ROOT TTree plugin per file.

Spec: "ROOT-free AO2D Reader Plugin — Implementation Spec",
https://claude.ai/artifact/SVBE64S6nZozEmFnAf4CXf (milestones, byte layouts,
I/O and memory strategy, DPL integration). These notes record what was
implemented against it, where it deviates, and what is verified.

## Status by milestone

| # | Milestone | State |
|---|---|---|
| 1 | Inspect tool: local ByteSource, container and object layers | Done, verified |
| 2 | Decode to heap Arrow: codecs, basket decoder, `o2-aod-native-dump` | Done, verified |
| 3 | Bounded memory | Done, verified (see the memory model notes) |
| 3b | `AODInputFile` abstraction in DataInputDirector | Done, existing tests pass |
| 4 | DPL plugin, `native\|root\|auto` selection, per-file fallback | Done, byte-identical to the TTree plugin on production AO2Ds, identical DPL results |
| 5 | Remote I/O: XRootD, HTTP, AliEn, read planning, prefetch, streaming | Done for XRootD, AliEn and public HTTP(S); authenticated HTTPS and S3 not started |
| 6 | Hardening, default switched to `auto` | Not started |

"Verified" means against the uproot-written corpus and, since milestone 4,
byte-identical to the TTree plugin on production AO2Ds (see below).

## Layout

```
Framework/AODNative/                O2::AODNative, ROOT-free (CMake fails if it links ROOT or O2::Framework)
  include/AODNative/
    Errors.h        FormatError (base), UnsupportedVersion, UnsupportedLayout, CorruptData, IoError
    ByteSource.h    Options, ReadRequest, ByteSource (+ willNeed/dontNeed hints)
    Container.h     FileHeader, KeyInfo, RootFile (keys, directories, DF ordering, payloads)
    Objects.h       BasketRef, BranchInfo, TreeInfo, supported class versions
    Codecs.h        PayloadDecompressor (incremental + prefix reads), decompressBounded, xxh64
    Decoder.h       BranchDecoder: values, VLA values, booleans, offsets, into caller memory
    ArrowTable.h    physicalSchema/datasetSchema (TTreePlugin contract), ColumnOp planning, makeBatch
    ReadPlan.h      ReadPlan (per-DF coalesced reads), ReadBufferPool, PlannedByteSource
  src/              LocalByteSource (mmap / pread), XRootDByteSource (XrdCl, if found),
                    HttpByteSource (libcurl), Cursor (bounds-checked BE reader), Endian, ...
  tools/            o2-aod-native-inspect [--json], o2-aod-native-dump <file> <DF/tree> <out.arrow>
  test/             Catch2 unit tests (o2-test-framework-aod-native)
    test_Memory.cxx   o2-test-framework-aod-native-memory <files>: private memory per DF (malloc interposed)
    fuzz/             libFuzzer target, -DAODNATIVE_FUZZ=ON (clang)
    tools/            make_corpus.py, reference_inspect.py (uproot reference), check_corpus.py
    standalone/       CMake harness to build this directory without the O2 stack

Framework/AnalysisSupport/test/
  compareNativeToRoot.cxx  o2-test-framework-aod-native-vs-root <files>: every tree
                           through both plugins, serialised as DPL does, byte compared

Framework/AnalysisSupport/src/
  AODInputFile.{h,cxx}   AODInputFile interface + RootAODInputFile (TFile)      [3b]
  NativeAOD.{h,cxx}      NativeAODFileSystem, NativeAODInputFile, NativeTreeHandle [4]
  NativeAODPlugin.cxx    FrameworkAnalysisNativeAODSupport: format, fragment,
                         deferred output stream, implementation and capability  [4]
```

Other changes: `DataInputDirector.{h,cxx}` and `Plugin.cxx` (DiscoverMetadataInAOD)
go through `AODInputFile`; `--aod-reader-backend`, `--aod-reader-threads`,
`--aod-reader-local-io` in `WorkflowHelpers.cxx` / `runDataProcessing.cxx`,
read in `AODJAlienReaderHelpers.cxx`; codec dependencies (ZLIB, and pkg-config
libzstd, liblz4 and liblzma) in `dependencies/O2Dependencies.cmake`. liblzma
goes through pkg-config too: `find_package(LibLZMA)` misses Homebrew's xz,
because the O2 recipe sets `CMAKE_IGNORE_PATH=/opt/homebrew/include`.

## Deviations from the spec

- Byte layouts, checked against uproot 5.7:
  - TObject's referenced bit is `kIsReferenced = 0x10`, not `0x00010000`.
  - TBranch v13 `fBasketSeek` is always 64-bit (`Q`); the flag byte is a speedbump.
  - Embedded baskets are parsed using ROOT's TBasket flag byte (99 header only,
    +10 buffer present, +40 displacements, 1 = entry offsets present).
  - Pure object back-references have no byte count: the first word is the tag.
- XXH64 for the LZ4 checksum is implemented in `Codecs.cxx` (about 50 lines):
  alidist has no xxhash package. zstd comes from the system via pkg-config:
  alidist has no zstd package either.
- `IoError` derives from `FormatError` like the other errors, so in `auto`
  mode unsupported URL schemes fall back to ROOT.
- `AODInputFile::topLevelKeys()` returns name + class pairs and there is an extra
  `keysIn(dir)`: `getListOfTables` (metadata discovery) needs both. TMap
  contents come back as `std::map`, so metadata is now iterated alphabetically.
- The native `RootObjectReadingCapability` lives in the plugin library
  (`O2FrameworkAnalysisNativeAODSupport:NativeAODObjectReadingCapability`), not
  in `Framework/Core/src/Plugin.cxx`: recognising a native handle needs
  `NativeAODFileSystem`, which Core cannot depend on. It is loaded (first in the
  list) only when `--aod-reader-backend` is not `root`, so the default path is
  unchanged.
- `NativeAODInputFile::open` parses the trees of the first DF, so that version
  or layout problems surface at open, when the file can still go to ROOT.
  Problems in later DFs become `AODReadError{InvalidTimeframe}`. `CorruptData`
  found inside the deferred `Write()` cannot be mapped (same as the TTree plugin).
- `TAttLine`, `TAttFill` and `TAttMarker` are skipped by byte count without
  a version check: ROOT 6.36 writes `TAttMarker` v3, and nothing is read from them.
- VLA sizes come from the `<name>_size` branch preceding the values, as in the
  TTree plugin, not from the leaf count: Run 2 converted FV0/FT0 tables have
  `fAmplitudeA` counted by `fChannelA_size` but preceded by `fAmplitudeA_size`.
- Metadata objects are refused above 256 MiB, and every length read from the
  file is checked against the file size before allocating (found by fuzzing).

## Memory model as measured (milestone 3)

`o2-test-framework-aod-native-memory` (Linux only: it interposes glibc's malloc) decodes into MAP_SHARED memory (standing
in for shm) and tracks every heap allocation, including the codecs'.

- Value and VLA columns: no basket-sized private buffer. zstd decodes partial
  chunks with the buffer-less API straight into the destination (its history),
  with a fixed 256 KiB scratch for the last blocks. Largest allocation:
  zstd 0.25 MiB, zlib 0.03 MiB, LZ4 / uncompressed 0. Baskets of 402 MB decode
  with 0.09 MiB private.
- liblzma needs its dictionary per worker (8 MiB at the presets ROOT uses).
- Booleans are not bounded by the 64 KiB window alone: their output is
  bit-packed, so it cannot be the codec's history, and zstd / LZ4 keep up to one
  window / chunk (0.39 MiB in the corpus). The test reports them separately.
- pread mode holds the compressed bytes of the DF being read (the read plan,
  below); mmap mode holds nothing.

## Verification done

- Unit tests: 26 test cases (codecs incl. bounded/prefix reads at block and
  chunk boundaries, truncation, corruption, LZ4 checksum; cursor bounds;
  decoder with poisoned entry-offset tails and guard bytes, booleans across
  basket and window boundaries, VLA, embedded and uncompressed baskets, threads).
- Corpus (make_corpus.py): zstd, zlib, lzma, lz4 and uncompressed files, 3 DFs
  each in non-lexicographic order, 3 tables, every leaf type, fixed arrays (also
  of booleans), VLAs, baskets of 0 to 70001 entries, incompressible baskets
  stored uncompressed; plus a 2.5 GB zstd file (64-bit seeks, 440 MB baskets
  over many 16 MiB chunks) and an RNTuple file. check_corpus.py: inspect JSON
  and dumped Arrow tables identical to the uproot reference for all 45 trees and
  the big file, IPC bytes independent of `--threads 4` / `--pread`, also under
  ASan; clean refusals for RNTuple, a patched TBranch version, truncations and
  non-ROOT files.
- Fuzzing: about 12 million inputs in total, ASan + UBSan (fatal); no open findings.

Not covered by the corpus: `metaData` / `parentFiles` TMaps and embedded
baskets (uproot writes neither).

- Production AO2Ds (`o2-test-framework-aod-native-vs-root`, macOS arm64): the
  IPC bytes DPL would send are identical between the native reader and the
  TTree plugin for every tree, with mmap and with `--threads 4 --pread`;
  `parentFiles` / `metaData` and DF keys identical too:
  - two Hyperloop derived files, ROOT 6.36, zstd: 264 DFs / 792 trees each
  - a Hyperloop derived file, ROOT 6.32, zstd, 937 MB: 27 DFs / 567 trees
  - a Run 2 converted AO2D, ROOT 6.24, zlib, 1.3 GB: 16 DFs / 384 trees
  In total 7.2 GB serialised. Truncated files are refused with `CorruptData`.
- `test_DataInputDirector`, `test_NavigateToLevel` (default `root` backend) pass.
- LHC24ar apass1 AO2D (ROOT 6.32, 2.6 GB): 63 DFs / 1827 trees byte-identical.
- DPL: `o2-testworkflows-analysis-histograms` on that file (it picks the
  `TracksIU` task from the discovered `aod-metadata-tables`): AnalysisResults.root
  identical, all 5 histograms bin by bin, with `--aod-reader-backend` root,
  native and auto.
- Fallback: 3 DFs of that file plus a tree with a `std::vector<float>` branch
  in the first DF (a `TBranchElement`, refused natively, never read by the
  workflow). `auto` logs the fallback once with the reason and gives results
  identical to `root`; `native` stops with the reason in the error.
- Not yet exercised in DPL: parent files, and the value of the TF file name
  sent with derived output.

## Read performance (2026-10-28)

`o2-test-framework-aod-native-vs-root --bench root|native <file>`: every tree of
every DF read and serialised as DPL does, one backend per process. macOS arm64,
warm page cache, median of 3 runs (spread about 1%). The reader alone, not DPL.

| File | Backend | Wall s | User s | Sys s | Max RSS MiB | Read calls |
|---|---|---|---|---|---|---|
| Hyperloop derived, 6.36, zstd, 90 MB | root | 0.45 | 0.42 | 0.02 | 273 | 2379 |
| | native | 0.36 | 0.35 | 0.01 | 327 | 5023 |
| | native --threads 4 | 0.35 | 0.35 | 0.01 | 326 | 5023 |
| | native --pread | 0.35 | 0.34 | 0.01 | 241 | 5023 |
| Hyperloop derived, 6.32, zstd, 937 MB | root | 3.98 | 2.64 | 1.34 | 399 | 1164 |
| | native | 2.58 | 2.50 | 0.07 | 1204 | 86424 |
| | native --threads 4 | 0.90 | 2.74 | 0.26 | 1204 | 86424 |
| | native --pread | 2.61 | 2.47 | 0.14 | 310 | 86424 |
| Run 2 converted, 6.24, zlib, 1.3 GB | root | 2.99 | 2.69 | 0.30 | 519 | 789 |
| | native | 2.67 | 2.58 | 0.08 | 1589 | 4191 |
| | native --threads 4 | 2.10 | 2.62 | 0.10 | 1589 | 4191 |
| | native --pread | 2.72 | 2.58 | 0.13 | 365 | 4191 |

- User time is the same (decompression dominates); the gain is ROOT's system
  time (1.34 s on the 937 MB file, not profiled).
- Threads split the baskets of one branch, so only files with many baskets per
  branch gain.
- With mmap, max RSS includes the mapped file. `madvise(MADV_DONTNEED)` after
  each decoded basket was tried: on macOS it changes nothing (Darwin keeps
  file-backed pages resident) and adds 0.1-0.3 s of system time, so it was not
  kept. On Linux it should drop the pages of the read-only `MAP_PRIVATE`
  mapping; to be measured there. pread is as fast and has the lowest RSS, so
  it is the DPL default (`--aod-reader-local-io pread`); the library and the
  tools keep mmap as their default, with `--pread` to switch.
- Many more read calls than ROOT (no TTreeCache): free locally, but remote
  reading needs the coalesced per-DF read plan of milestone 5 (next section).

## Read plan (milestone 5, step 1)

DPL decodes each tree as soon as it is read (`f2b.release()` in
`DataInputDescriptor::readTree` serialises right away), so the plan cannot wait
for all the tables of a DF. Instead it learns, like TTreeCache: a reader asks
for the same tables in every DF.

- `NativeAODFileSystem::planFor(path)`: when a tree of a new DF is asked for, a
  new `ReadPlan` is made with every basket of the trees read in the previous DF
  registered, plus this tree. The first lookup of a range not in memory fetches
  all pending ranges with one `readvInto`, merging ranges less than
  `Options::coalesceGap` (64 KiB) apart. The first DF of a file is read tree by tree.
- Each fragment decodes through a `PlannedByteSource` over the plan. With mmap
  there is no plan: nothing needs reading.
- Fetch buffers come from a `ReadBufferPool` per file and go back to it when
  the plan goes away, so DFs reuse one buffer instead of allocating and
  faulting in a new one each time.

Byte-identical on the 5 production files with `--pread` and `--pread --threads 4`;
DPL results identical. Read calls reading every tree (per basket before / with
the plan / ROOT): hy_3993289/0065 5023 / 2121 / 2379; hy_1318738 86424 / 1232 /
1164; AO2D_019 4191 / 497 / 789; LHC24ar 71799 / 2273 / 3783. Bytes read +0.07%
(merged gaps). In DPL (`analysis-histograms` on LHC24ar, a few tables):
8852 → 1209 read calls, ROOT 444; about 19 merged ranges per DF, because the
requested baskets are interleaved with those of unread trees. They go out in one
`readvInto` per DF, which is one vectored request for a remote source; locally
it is still one pread each.

Cost, locally: wall time within 0-2% of pread without the plan. Max RSS grows by
about one DF of compressed bytes (macOS, `MallocLargeCache=0`: hy_1318738
310 → 349 MiB, AO2D_019 364 → 444, LHC24ar 311 → 360), the same order as
ROOT's 50 MB TTreeCache. With the macOS allocator cache on, max RSS is higher
(378, 599, 435 MiB): it keeps freed large blocks resident; glibc returns them.

## XRootD source (milestone 5, step 2)

`XRootDByteSource` (root://, roots://, xroot://, xroots://), built when O2 finds
XRootD (optional in O2); otherwise those URLs get an IoError, so `auto` falls
back. Links only libXrdCl. `readvInto` sends a whole read plan at once: ranges
up to 2097136 bytes go in vectored reads of at most 1024 chunks, larger ones
in plain reads of at most 64 MiB, all asynchronous, then waits for all of them;
a short or failed reply is an IoError. `readCalls()` counts requests sent.

Tested against a local server (Homebrew xrootd 6.2.0, `all.export ~/src/data
r/o`): byte-identical with both readers over root:// on hy_3993289/0065,
hy_1318738 and LHC24ar (7.8 GB). Over localhost, median of 3 (wall s, requests):

| File | root | native | native --threads 4 |
|---|---|---|---|
| hy_3993289/0065 | 0.90, 2379 | 0.72, 2121 | 0.72 |
| hy_1318738 | 3.45, 1164 | 3.31, 727 | 1.36 |
| LHC24ar | 7.73, 3783 | 7.30, 2175 | 3.11 |

No latency on localhost, so the request count is what transfers to a real
network. Of native's requests, the data takes 3-4 per DF; the rest are
metadata read one at a time (directory header, key list, one per TTree).

Metadata, batched (step 3):
- `RootFile::keysOf(df)` reads the key list of a DF once (it was read twice
  per DF: once by `NativeAODInputFile::keysIn` and once by the filesystem) and
  fetches the stored bytes of all its TTrees with it, in one `readvInto`:
  ranges less than 4 KiB apart are merged (64 KiB pulled basket data in, +16%
  bytes on small DFs). They are parsed only when asked for, so a TTree the
  native reader cannot parse still fails only if requested. About 30 kB per
  DF on LHC24ar and hy_1318738, 2 kB on hy_3993289.
- The entry count of a last basket missing from `fBasketEntry` is computed
  from the branch's entries; the decoder checks it against the basket header.
  The header read stays as fallback.

Requests per DF are now 4 sequential round trips: directory record, key list,
TTrees, data (a few requests in flight). Over root://localhost, every tree,
median of 3 (wall s, requests):

| File | root | native | native --threads 4 |
|---|---|---|---|
| hy_3993289/0065 | 0.89, 2379 | 0.56, 1063 | 0.57 |
| hy_1318738 | 3.49, 1164 | 3.23, 131 | 1.25 |
| LHC24ar | 7.77, 3783 | 7.09, 283 | 2.84 |

2.2x to 13x fewer requests than ROOT, bytes read within 0.1%.

Only the TTrees the read plan learned (those read in the previous DF, set with
`RootFile::setTreesToPrefetch`) are prefetched; all of them for the first DF of
a file, none when the file is mapped. Fetching every TTree of the DF cost, in
DPL (analysis-histograms, LHC24ar, a few tables), 2564 preads instead of 1209
and +0.3% bytes; with the learned set it is 1224 preads and the same bytes as
before. Over root:// the same workflow sends 319 requests (ROOT 444), with
identical results.

## Next-DF prefetch (milestone 5, step 4)

For a remote source (`ByteSource::isRemote()`, XRootD) and `prefetchDepth > 0`
(default 1), when a DF starts `NativeAODFileSystem` prepares the next one in a
background task: its keys, the TTrees learned in the current DF, and a read
plan with their baskets fetched (`ReadPlan::fetch()`). It works on its own copy
of `RootFile` over the same source, so `RootFile` is never shared across
threads. When the reader reaches that DF, the keys, trees and plan are adopted;
a lookahead for another DF, or one which failed, is dropped (the normal path
redoes the work and reports errors). Not started for the first DF of a file,
which has not shown yet which trees are read.

The next DF is the one after the current in the read order DPL uses:
`DataInputDescriptor` passes its `listOfTimeFrameNumbers` through
`AODInputFile::setReadOrder()`. `RootFile::dataFrames()` could not stand in for
it: with a parentFiles map, DataInputDirector sorts by parent file with an
unstable sort, so DFs of the same parent can come in a different order.

Also: `readStringMap()` results are cached (parentFiles was read twice at open).

Over root://localhost, every tree, median of 3 (wall s):

| File | root | native, no prefetch | native | --threads 4: no prefetch / prefetch |
|---|---|---|---|---|
| hy_3993289/0065 | 0.89 | 0.56 | 0.36 | 0.56 / 0.36 |
| hy_1318738 | 3.43 | 3.24 | 2.61 | 1.24 / 0.92 |
| LHC24ar | 7.75 | 7.11 | 5.38 | 2.91 / 2.30 |

Over root:// native now runs at the speed of local pread (LHC24ar 5.48 s).
Cost: one more DF of compressed bytes (max RSS +34 MiB on hy_1318738, +81 MiB on
LHC24ar). Requests are about the same as without it (151 vs 130, 283 vs 283);
in DPL (analysis-histograms, LHC24ar) 380 vs 319, with identical results.

## AliEn lookup (milestone 5, step 5)

`alien://` files: `gridReplicas(lfn)` (AODInputFile.cxx, which may use ROOT)
sends `access read <lfn>` through `gGrid`, as `TJAlienFile::Open` does, and
turns each replica into `root://<se>//<pfn>?&authz=<envelope>`. Files inside
an archive (`#member`, on the LFN or the replica) are refused, so `auto` falls
back to ROOT. `openNativeAOD()` tries the replicas in the order AliEn proposes,
moving on after an IoError only (a format problem is the same on every
replica); DPL's reader and the comparison tool both open files through it.
The file keeps its LFN as name: what DataInputDirector compares, what goes
with derived output, and what monitoring reports, with `se=` and
`open_time=` as for TJAlienFile. `XRootDByteSource` uses the full URL only to
open; its `url()` and error messages leave out the query, so the access
envelope does not reach logs or outputs.

From a Mac at CERN, token from alien-token-init:
- hy_3993289/0065 (alien:///alice/cern.ch/user/a/alihyperloop/jobs/0399/...):
  byte-identical with ROOT (TJAlienFile), 264 DFs / 792 trees.
- LHC24ar, first 10 DFs, `--bench`: root 25.15 s, 603 requests, 2.70 s user,
  382 MiB; native 19.93 s, 71 requests, 1.25 s user, 522 MiB. Both read about
  408 MB: bandwidth bound (16 vs 20 MB/s from CERN EOS), native keeps the
  connection busier thanks to the lookahead.
- DPL, `o2-testworkflows-analysis-histograms --aod-file alien://.../LHC24ar/...AO2D.root`:
  AnalysisResults.root identical (5 histograms) with root, native and auto.
  Requests 444 (root) vs 380 (native), 628.5 vs 629.1 MB read. The open
  line reads `lfn=alien://...,size=2630072561,se=ALICE::CERN::EOS,open_time=0.1`,
  and no `authz` appears in the log. Wall times (root 49.4 s, native 48.7 s,
  auto 35.2 s with the same native code) vary by 13 s from run to run over the
  WAN: a timing comparison needs interleaved repeats, or a Hyperloop test.

## HTTP(S) source (milestone 5, step 6)

`HttpByteSource` (http://, https://), libcurl, public servers only (no
authentication yet). The size comes from a HEAD request; reads are ranged GETs,
ranges above 16 MiB split so that the pieces go out in parallel. `readvInto`
sends all its GETs at once on a curl multi handle: multiplexed on one
connection with HTTP/2 (`CURL_HTTP_VERSION_2TLS` by default, `httpVersion`),
otherwise over at most `httpConnections` (4) connections. Connections, TLS
sessions and DNS are shared by all the calls on a source (a curl share
handle), so the lookahead and the reader can fetch at the same time.
Redirects are followed; `X509_CERT_DIR` sets the CA path. A reply must be a 206
with exactly the bytes asked for: a server which ignores the range is caught at
the first extra byte and refused ("the server ignored the byte range"), so
`auto` falls back to ROOT rather than downloading the whole file.

Tested: `o2-aod-native-inspect https://root.cern/files/h1/dstarmb.root` (HEAD and
ranges over public HTTPS); Python's http.server, which ignores ranges, is
refused. Against a local range-capable test server (HTTP/1.1, single ranges),
native over HTTP vs ROOT on the local file (the tool takes
`<native url>=<ROOT url>`): byte-identical on hy_3993289/0065, hy_1318738 and
LHC24ar, also with --threads 4. hy_1318738, localhost: 3.14 s without
prefetch, 2.65 s with, 0.97 s with --threads 4 (root:// 3.24, 2.61, 0.92).

Public HTTP, alimonitor train test data (LHC15o Run 2 converted, ROOT 6.24,
zlib, 1.19 GB, http://alimonitor.cern.ch/train-workdir/testdata/LFN/alice/data/2015/LHC15o/000246392/pass2/PWGZZ/Run3_Conversion/320_20220701-0813/AOD/153/AO2D.root):
byte-identical with ROOT on a local copy (15 DFs, 360 trees, 3.7 GB
serialised). Bandwidth bound at about 26 MB/s, as wget (44.0 s for the file):
native 47.8 s without prefetch, 45.0 s with, 46.1 s with --threads 4; 3.2 s of
user CPU. This ROOT build cannot read http:// at all (the TWebFile plugin entry
fails in ROOT 6.40, and there is no Davix), so for these files `auto` has no
fallback and the native reader is the only way.

## Building and testing

In O2: built with the rest of Framework (aliBuild). Standalone, for iterating on
the core without the O2 stack:

```sh
python3 -m venv .venv && .venv/bin/pip install cmake ninja uproot awkward numpy pyarrow xxhash
curl -sSLO https://raw.githubusercontent.com/catchorg/Catch2/v3.7.1/extras/catch_amalgamated.{hpp,cpp}   # into $CATCH2
PATH=$PWD/.venv/bin:$PATH cmake -S O2/Framework/AODNative/test/standalone -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPYARROW_DIR=$PWD/.venv/lib/python3.X/site-packages/pyarrow -DCATCH2_DIR=$CATCH2
ninja -C build
# add -DSANITIZE=ON for ASan/UBSan; CXX=clang++ -DSANITIZE=ON -DAODNATIVE_FUZZ=ON for the fuzzer
build/tests/o2-test-framework-aod-native
.venv/bin/python -I O2/Framework/AODNative/test/tools/make_corpus.py corpus [--big]
.venv/bin/python -I O2/Framework/AODNative/test/tools/check_corpus.py build/bin corpus work
build/tests/o2-test-framework-aod-native-memory [--threads N] [--pread] corpus/*.root
```

## Next steps

1. alidist (`o2.sh` needs `lzma` and `lz4`; zstd from the system) and the
   Linux build, with the memory test and `madvise` measured there.
2. A Hyperloop test train with `--aod-reader-backend auto` on the reader,
   including merged files with embedded baskets and derived data with parent files.
3. Authenticated HTTPS (tokens, X.509) and S3, when needed.

Build note for Arch Linux (kernel headers 7.x): alidist's Clang 20.1.7 fails in
compiler-rt on `<linux/scc.h>`; adding `-DCOMPILER_RT_BUILD_SANITIZERS=OFF` to
the cmake call in `alidist/clang.sh` works around it.
