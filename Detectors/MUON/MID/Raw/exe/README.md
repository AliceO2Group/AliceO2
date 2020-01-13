<!-- doxy
\page refMUONMIDRawExe MID RAW executable
/doxy -->

# Utilities for MID RAW data
This directory contains two utilities that can be executed to verify the raw data.
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

## MID raw file checker
This utility allows to test files produced by the CRU w/o using the User Logic.
Basic usage:
```bash
o2-mid-raw-checker filename [filename_2 filename_3 ...]
```
The output is a file (or a series of files if a list of inputs is provided), named after the input file as `check_filename.txt`, so that it is easy to understand to which input file the output is referring to.
The different e-links are read out and then grouped according to their interaction record.
The output file contains:
-   a list of possible problems (if any) for each event processed

-   a summary with the number of faulty events
The list of problems is preceded by the identifier of the processed HB and the corresponding "line" in the file where the HB is.
The line is counted assuming that one reads the file as a series of 4 words of 32 bits each (this is the typical way the binary file is converted into text during the tests).
Notice that one can have more than 1 HB per event, since the data might be split into different pages if the maximum page size is reached.
For a list of the other available options:
```bash
o2-mid-raw-checker --help
```
### Performed checks
The decoded information is read HB per HB (multiple pages are read out when needed).
For each HB, the decoded information are gathered according to their interaction.
Notice that, in principle, events belonging to different HBs should have different interaction records, so one could in principle read the full file and then perform the check.
However, in the tests the RDH is not always correctly set, and the orbit number might not increase. That is why we read the HB one by one, and limit the tests to 1 HB at the time. This makes it easier to tag HBs with issues in the RO.
For each interaction record, a number of checks are performed:
-   The number of local cards read must be 4 times the number of regional cards. Notice that this error often implies that there is a mismatch between the regional and local clocks (so the cards gets split into different interaction records)
-   The word describing the event (SOX, EOX, HC, Calibration, etc.) should be the same for all cards in the interaction record.
-   The number of non-zero patterns read must match the information written in the corresponding word that indicates the non-zero detector planes. Notice that this might not be true when the patterns represent the masks, since in this case it is fine to transmit a zero pattern. However, for the sake of simplicity, we do not account for this possibility. Of course, this implies that must run the check on tests where all masks are non-zero.
-   For each card we check that the information is consistent. For example we cannot have SOX and Calibration bits fired at the same time. Also, during an HC, we expect to have no chamber fired for the local boards. Notice that during tests this information is added by hand, so we should not have any issue by design.