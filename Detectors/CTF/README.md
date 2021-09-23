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
o2-its-reco-workflow --entropy-encoding | o2-ctf-writer-workflow --onlyDet ITS
```

For the storage optimization reason one can request multiple CTFs stored in the same output file (as entries of the `ctf` tree):
```bash
o2-ctf-writer --min-file-size <min> --max-file-size <max> ...
```
will accumulate CTFs in entries of the same tree/file until its size fits exceeds `min` and does not exceed `max` (`max` check is disabled if `max<=min`) or EOS received.
The `--max-file-size` limit will be ignored if the very first CTF already exceeds it.

The output directory (by default: `cwd`) for CTFs can be set via `--output-dir` option and must exist. Since in on the EPNs we may store the CTFs on the RAM disk of limited capacity, one can indicate the fall-back storage via `--output-dir-alt` option. The writer will switch to it if
(i) `szCheck = max(min-file-size*1.1, max-file-size)` is positive and (ii) estimated (accounting for eventual other CTFs files written concurrently) available space on the primary storage is below the `szCheck`. The available space is estimated as:
````
current physically available space
-
number still open CTF files from concurrent writers * szCheck
+
the current size of these files
````

Option `--ctf-dict-dir <dir>` can be provided to indicate the (existing) directory for the dictionary IO.

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
--ctf-input arg (=none)
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

For the ITS and MFT entropy decoding one can request either to decompose clusters to digits and send them instead of clusters (via `o2-ctf-reader-workflow` global options `--its-digits` and `--mft-digits` respectively)
or to apply the noise mask to decoded clusters (or decoded digits). If the masking (e.g. via option `--its-entropy-decoder " --mask-noise "`) is requested, user should provide to the entropy decoder the noise mask file (eventually will be loaded from CCDB) and cluster patterns decoding dictionary (if the clusters were encoded with patterns IDs).
For example,
```
o2-ctf-reader-workflow --ctf-input <ctfFiles> --onlyDet ITS,MFT --its-entropy-decoder ' --mask-noise --noise-file its_noise.root --cluster-dict-file ./ ' | ...
```
will decode ITS and MFT data, decompose on the fly ITS clusters to digits, mask the noisy pixels with the provided masks, recluster remaining ITS digits and send the new clusters out, together with unchanged MFT clusters.
```
o2-ctf-reader-workflow --ctf-input <ctfFiles> --onlyDet ITS,MFT --mft-digits --mft-entropy-decoder ' --mask-noise --noise-file mft_noise.root --cluster-dict-file ./ ' | ...
```
will send decompose clusters to digits and send ben out after masking the noise for the MFT, while ITS clusters will be sent as decoded.


## Support for externally provided encoding dictionaries

By default encoding with generate for every TF and store in the CTF the dictionary information necessary to decode the CTF.
Since the time needed for the creation of dictionary and encoder/decoder may exceed encoding/decoding time, there is a possibility
to create in a separate pass a dictionary stored in the CTF-like object and use it for further encoding/decoding.

To create a dictionary run usual CTF creation chain but with extra option, e.g.:
```bash
o2-its-reco-workflow --entropy-encoding | o2-ctf-writer-workflow --output-type dict --onlyDet ITS
```
This will create a file `ctf_dictionary.root` containing dictionary data for all detectors processed by the `o2-ctf-writer-workflow`.
By default the dictionary file is written on the exit from the workflow, in `CTFWriterSpec::endOfStream()` which is currently not called if the workflow is stopped
by `ctrl-C`. Periodic incremental saving of so-far accumulated dictionary data during processing can be triggered by providing an option
``--save-dict-after <N>``.

Following encoding / decoding will use external dictionaries automatically if this file is found in the working directory (eventually it will be provided via CCDB).
Note that if the file is found but dictionary data for some detector participating in the workflow are not found, an error will be printed and for given detector
the workflows will use in-ctf dictionaries.
The dictionaries must be provided for decoding of CTF data encoded using external dictionaries (otherwise an exception will be thrown).

When decoding CTF containing dictionary data (i.e. encoded w/o external dictionaries), the CTF-specific dictionary will be created/used on the fly, ignoring eventually provided external dictionary data.
