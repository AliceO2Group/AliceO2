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

## CTF reader workflow

`o2-ctf-reader-workflow` should be the 1st workflow in the piped chain of CTF processing.
At the moment accepts as an input a comma-separated list of CTF files produced by the `o2-ctf-writer-workflow`, reads data for all detectors present in it
(the list can be narrowed by `--onlyDet arg (=none)` and `--skipDet arg (=none)` comma-separated lists), decode them using decoder provided
by detector and injects to DPL. In case of multiple entries in the CTF tree, they all will be read in row.

Example of usage:
```bash
o2-ctf-reader-workflow --onlyDet ITS --ctf-input o2_ctf_0000000000.root  | o2-its-reco-workflow --trackerCA --clusters-from-upstream --disable-mc
```

With `--delay <s>` a delay of `s` seconds will be introduced between injections of consecutive CTFs (if >1).
One can loop over the input by providing `--loop <N=1>` option.


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
