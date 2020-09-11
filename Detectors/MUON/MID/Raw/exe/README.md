<!-- doxy
\page refMUONMIDRawExe MID RAW executable
/doxy -->

# Utilities for MID RAW data
This directory contains utilities that can be executed to test the raw data.
This is particularly important for testing and debugging the MID Read-out.

## MID raw dumper
This utility allows to dump the decoded raw data content.
Basic usage:
```bash
o2-mid-rawdump filename [filename_2 filename_3 ...]
```
The output is a file containing a list with:
-   the interaction record

-   the list of decoded [LocalBoardRO](../include/MIDRaw/LocalBoardRO.h)
For a list of the other available options:
```bash
o2-mid-rawdump --help
```
