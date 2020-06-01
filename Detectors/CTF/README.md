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

## CTF reader workflow

`o2-ctf-reader-workflow` should be the 1st workflow in the piped chain of CTF processing.
At the moment accepts as an input a single file produced by the `o2-ctf-writer-workflow`, reads data for all detectors present in it
(the list can be narrowd by `--onlyDet arg (=none)` and `--skipDet arg (=none)` comma-separated lists), decode them using decoder provided
by detector and injects to DPL.

Example of usage:
```bash
o2-ctf-reader-workflow --onlyDet ITS --ctf-input o2_ctf_0000000000.root  | o2-its-reco-workflow --trackerCA --clusters-from-upstream --disable-mc
```
