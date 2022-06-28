<!-- doxy
\page refDetectorsCTF CTF Data handling
/doxy -->

# CTF I/O

## CTF writer workflow

`o2-ctf-writer-workflow` can be piped in the end of the processing workflow to create CTF data.
By default data of every detector flagged in the GRP as being read-out are expected. The list of detectors storing CTF data can be managed using
`--onlyDet arg (=none)` and `--skipDet arg (=none)` comma-separated lists.
Every detector writing CTF data is expected to send an output with entropy-compressed `EncodedBlocks` flat object.

Example of usage:
```bash
o2-its-reco-workflow | o2-itsmft-entropy-encoder-workflow | o2-ctf-writer-workflow --onlyDet ITS
```

For the storage optimization reason one can request multiple CTFs stored in the same output file (as entries of the `ctf` tree):
```bash
o2-ctf-writer-workflow --min-file-size <min> --max-file-size <max> ...
```
will accumulate CTFs in entries of the same tree/file until its size fits exceeds `min` and does not exceed `max` (`max` check is disabled if `max<=min`) or EOS received.
The `--max-file-size` limit will be ignored if the very first CTF already exceeds it.
Additional option `--max-ctf-per-file <N>` will forbid writing more than `N` CTFs to single file (provided `N>0`) even if the `min-file-size` is not reached. User may request autosaving of CTFs accumulated in the file after every `N` TFs processed by passing an option `--save-ctf-after <N>`.

The output directory (by default: `cwd`) for CTFs can be set via `--output-dir` option and must exist. Since in on the EPNs we may store the CTFs on the RAM disk of limited capacity, one can indicate the fall-back storage via `--output-dir-alt` option. The writer will switch to it if
(i) `szCheck = max(min-file-size*1.1, max-file-size)` is positive and (ii) estimated (accounting for eventual other CTFs files written concurrently) available space on the primary storage is below the `szCheck`. The available space is estimated as:
````
current physically available space
-
number still open CTF files from concurrent writers * szCheck
+
the current size of these files
````

If the option `--meta-output-dir <dir>` is not `/dev/null`, the CTF `meta-info` files will be written to this directory (which must exist!).

By default only CTFs will written. If the upstream entropy compression is performed w/o external dictionaries, then the for every CTF its own dictionary will be generated and stored in the CTF. In this mode one can request creation of dictionary file (or dictionary file per detector if option `--dict-per-det` is provided) by passing option `--output-type dict` (in which case only the dictionares will be stored but not the CTFs) or
`--output-type both` (will store both dictionaries and CTF). This is the only valid mode for dictionaries creation (if one requests dictionary creation but the compression was done with external dictionaries, the newly created dictionaries will be empty).
In the dictionaries creation mode their data are accumulated over all CTFs procssed. User may request periodic (and incremental) saving of dictionaries after every `N` TFs processed by passing `--save-dict-after <N>` option.

Option `--ctf-dict-dir <dir>` can be provided to indicate the directory where the dictionary will be stored.

The external dictionaries created by the `o2-ctf-writer-workflow` containes a TTree (one for all participating detectos) and separate dictionaries per detector which can be uploaded to the CCDB. The per-detector dictionaries compatible with CCDB can be also extracted from the common TTree-based dictionary file using the macro `O2/Detectors/CTF/utils/CTFdict2CCDBfiles.C` (installed to $O2_ROOT/share/macro/CTFdict2CCDBfiles.C) which extracts the dictionary for every detector into separate file containing plain `vector<char>`. These per-detector files can be directly
uploaded to CCDB and accessed via `CcdbAPI` (the reference of the vector should be provided to corresponding detector CTFCoder::createCoders method to build the run-time dictionary). These files can be also used as per-detector command-line
parameters, on the same footing as tree-based dictionaries, e.g.
```
o2-ctf-reader-workflow --ctf-input input.lst --onlyDet ITS,TPC,TOF --its-entropy-decoder ' --ctf-dict ctfdict_ITS_v1.0_1626472046.root' --tpc-entropy-decoder ' --ctf-dict ctfdict_TPC_v1.0_1626472048.root' --tof-entropy-decoder  ' --ctf-dict ctfdict_TOF_v1.0_1626472048.root'
```

See below for the details of `--ctf-dict` option.

## CTF reader workflow

`o2-ctf-reader-workflow` should be the 1st workflow in the piped chain of CTF processing.
At the moment accepts as an input a comma-separated list of CTF files produced by the `o2-ctf-writer-workflow`, reads data for all detectors present in it
(the list can be narrowed by `--onlyDet arg (=none)` and `--skipDet arg (=none)` comma-separated lists), decode them using decoder provided
by detector and injects to DPL. In case of multiple entries in the CTF tree, they all will be read in row.

Example of usage:
```bash
o2-ctf-reader-workflow --onlyDet ITS --ctf-input o2_ctf_0000000000.root  | o2-its-reco-workflow --trackerCA --clusters-from-upstream --disable-mc
```

The option are:
```
--ctf-input arg (=ccdb)
```
inptu data (obligatort): comma-separated list of CTF  files and/or files with list of data files and/or directories containing files

```
--onlyDet arg (=all)
```
comma-separated list of detectors to read, Overrides skipDet

```
--skipDet arg (=none)
```
comma-separated list of detectors to skip

```
--max-tf arg (=-1)
```
max CTFs to process (<= 0 : infinite)

```
--loop arg (=0)
```
loop N times after the 1st pass over the data (infinite for N<0)

```
--delay arg (=0)
```
delay in seconds between consecutive CTFs sending (depends also on file fetching speed)

```
--copy-cmd arg (=XrdSecPROTOCOL=sss,unix xrdcp -N root://eosaliceo2.cern.ch/?src ?dst)
```
copy command for remote files or `no-copy` to avoid copying

```
--ctf-file-regex arg (=.+o2_ctf_run.+\.root$)
```
regex string to identify CTF files: optional to filter data files (if the input contains directories, it will be used to avoid picking non-CTF files)

```
--remote-regex arg (=^/eos/aliceo2/.+)
```
regex string to identify remote files

```
--max-cached-files arg (=3)
```
max CTF files queued (copied for remote source).

There is a possibility to read remote root files directly, w/o caching them locally. For that one should:
1) provide the full URL the remote files, e.g. if the files are supposed to be accessed by `xrootd` (the `XrdSecPROTOCOL` and `XrdSecSSSKT` env. variables should be set up in advance), use
`root://eosaliceo2.cern.ch//eos/aliceo2/ls2data/...root` (use `xrdfs root://eosaliceo2.cern.ch ls -u <path>` to list full URL).
2) provide proper regex to define remote files, e.g. for the example above: `--remote-regex "^root://.+/eos/aliceo2/.+"`.
3) pass an option `--copy-cmd no-copy`.

```
--select-ctf-ids <id's of CTFs to select>
```
This is a `ctf-reader` device local option allowing selective reading of particular CTFs. It is useful when dealing with CTF files containing multiple TFs. The comma-separated list of increasing CTFs indices must be provided in the format parsed by the `RangeTokenizer<int>`, e.g. `1,4-6,...`.
Note that the index corresponds not to the entry of the TF in the CTF tree but to the reader own counter incremented throught all input files (e.g. if the 10 CTF files with 20 TFs each are provided for the input and the selection of TFs
`0,2,22,66` is provided, the reader will inject to the DPL the TFs at entries 0 and 2 from the 1st CTF file, entry 5 of the second file, entry 6 of the 3d and will finish the job.

For the ITS and MFT entropy decoding one can request either to decompose clusters to digits and send them instead of clusters (via `o2-ctf-reader-workflow` global options `--its-digits` and `--mft-digits` respectively)
or to apply the noise mask to decoded clusters (or decoded digits). If the masking (e.g. via option `--its-entropy-decoder " --mask-noise "`) is requested, user should provide to the entropy decoder the noise mask file (eventually will be loaded from CCDB) and cluster patterns decoding dictionary (if the clusters were encoded with patterns IDs).
For example,
```
o2-ctf-reader-workflow --ctf-input <ctfFiles> --onlyDet ITS,MFT --its-entropy-decoder ' --mask-noise' | ...
```
will decode ITS and MFT data, decompose on the fly ITS clusters to digits, mask the noisy pixels with the provided masks, recluster remaining ITS digits and send the new clusters out, together with unchanged MFT clusters.
```
o2-ctf-reader-workflow --ctf-input <ctfFiles> --onlyDet ITS,MFT --mft-digits --mft-entropy-decoder ' --mask-noise' | ...
```
will send decompose clusters to digits and send ben out after masking the noise for the MFT, while ITS clusters will be sent as decoded.

By default an exception will be thrown if detector is requested but missing in the CTF. To enable injection of the empty output in such case one should use option `--allow-missing-detectors`.

```
--ctf-data-subspec arg (=0)
```
allows to alter the `subSpecification` used to send the CTFDATA from the reader to decoders. Non-0 value must be used in case the data extracted by the CTF-reader should be processed and stored in new CTFs (in order to avoid clash of CTFDATA messages of the reader and writer).

## Support for externally provided encoding dictionaries

In absence of the external dictionary the encoding with generate for every TF and store in the CTF the dictionary information necessary to decode the CTF.
Since the time needed for the creation of dictionary and encoder/decoder may exceed encoding/decoding time, there is a possibility
to create in a separate pass a dictionary stored in the CTF-like object and use it for further encoding/decoding.

The option `--ctf-dict <OPT>` steers in all detectors entropy encoders the fething of the entropy dictionary. The choices for OPT are:
1) `"ccdb"` (or empty string): leads to using CCDB objec fetching by the DPL CCDB service (default)

2) `<filename>`: use the dictionary from provided file (either tree-based format or flat one in CCDB format)

3) `"none"`: do not use external dictionary, instead per-TF dictionaries will be stored in the CTF


To create a dictionary run usual CTF creation chain but with extra option, e.g.:
```bash

o2-its-reco-workflow | o2-itsmft-entropy-encoding-workflow --ctf-dict none | o2-ctf-writer-workflow --output-type dict --onlyDet ITS
```
This will create a file `ctf_dictionary_<date>_<NTF_used>.root` (linked to `ctf_dictionary.root`) containing dictionary data in a TTree format for all detectors processed by the `o2-ctf-writer-workflow`.
Additionally, for every participation detector a `ctf_dictionary_<DET>_v<version>_<data>_<NTF_used>.root` file will be produced, with the dictionary in the flat format. These files can be directly uploaded to the CCDB.
By default the dictionary file is written on the exit from the workflow, in `CTFWriterSpec::endOfStream()` which is currently not called if the workflow is stopped
by `ctrl-C`. Periodic incremental saving of so-far accumulated dictionary data during processing can be triggered by providing an option
``--save-dict-after <N>``.

When decoding CTF containing dictionary data (i.e. encoded w/o external dictionaries), externally provided dictionaries will be ignored.
