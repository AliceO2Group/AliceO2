<!-- doxy
\page refMUONMIDCalibrationExe MID Calibration executable
/doxy -->

# MID calibration

This directory contains code to handle the MID calibration

## MID bad channels CCDB

This utility allows to query or upload the bad channels from or to the CCDB.

To query:

```bash
o2-mid-bad-channels-ccdb -q -v -t <timestamp>
```

The `-v` option also prints the bad files.
`timestamp` is the timestamp from which one wants to query.

To upload the default list of bad channels (empty list):

```bash
o2-mid-bad-channels-ccdb -p -t 1
```

The defauult CCDB is: `http://ccdb-test.cern.ch:8080`.
