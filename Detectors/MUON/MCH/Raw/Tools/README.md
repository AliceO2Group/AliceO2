<!-- doxy
\page refDetectorsMUONMCHRawTools Tools
/doxy -->

# MCH RawData Tools

## Raw Dump

The `o2-mchraw-dump` executable reads events from a raw data file and computes
the mean and sigma of the data of each channel.

The data format (Bare or UserLogic) is deduced from the data itself.

Until the data is completely self-consistent though (when we'll move to RDH v5
or 6), you'll have to provide the CRU id with the `--cru` option.

```bash
> o2-mchraw-dump -h
Generic options:
  -h [ --help ]           produce help message
  -i [ --input-file ] arg input file name
  -n [ --nrdhs ] arg      number of RDHs to go through
  -s [ --showRDHs ]       show RDHs
  -j [ --json ]           output means and rms in json format
  --cru arg               force cruId

> o2-mchraw-dump -i $HOME/cernbox/o2muon/data-DE617.raw -j -n 10000 -cru 0
```

Note that options `-j` and `-s` are incompatible (as the rdh output is not in
json).
