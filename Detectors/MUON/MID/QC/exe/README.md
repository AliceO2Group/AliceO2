<!-- doxy
\page refMUONMIDQCExe MID QC executable
/doxy -->

# Checkers for MID RAW data
This directory contains executable code to verify the raw data.
This is particularly important for testing and debugging the MID read-out.

## MID raw file checker
This utility allows to test files produced by the CRU w/o using the User Logic.
Basic usage:
```bash
o2-mid-raw-checker --feeId-config-file feeId_filename filename [filename_2 filename_3 ...]
```
The `feeId_filename` is a file allowing to tell which feeId is readout by the configured GBT.
The file should be in the form explained [here](../../Raw/README.md)

The output is a file (or a series of files if a list of inputs is provided), named after the input file as `check_filename.txt`, so that it is easy to understand to which input file the output is referring to.
The different e-links are read out and then grouped according to their interaction record.
The output file contains:
-   a list of possible problems (if any) for each event processed
-   a summary with the number of faulty events

The list of problems is provided per interaction record.
This should be unique, but it is not always the case, since it can happen that either the decoded local clock information is wrong, or some reset is performed and the same interaction record is repeated.
The data corresponding to one interaction record can be found in different pages. Notice that this also happens for a perfectly good event in case the information does not fit inside the maximum page size.
For debugging purposes, we try to keep track of the page (HB) where the data corresponding to the interaction record were found.
This is rather accurate, although sometimes the data can be found in the preceding page.
We therefore print the interaction records and the corresponding pages (HB), together with the line in the file where the page start.
The line is counted assuming that one reads the file as a series of 4 words of 32 bits each (this is the typical way the binary file is converted into text during the tests).

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
-   The number of local cards read must be compatible with the number of non-null cards provided by the regional cards. Notice that a mismatch can occur when the local board is busy (but does not correctly signal it) and we therefore get the regional info but not the corresponding local card. An incompatibility can also appear in case of corrupted data reading.
-   The word describing the event (SOX, EOX, HC, Calibration, etc.) should be the same for all cards in the interaction record.
-   The number of non-zero patterns read must match the information written in the corresponding word that indicates the non-zero detector planes.
-   For each card we check that the information is consistent. For example we cannot have SOX and Calibration bits fired at the same time. Also, during an HC, we expect to have no chamber fired for the local boards. Notice that during tests this information is added by hand, so we should not have any issue by design.
-   When the overwritten bit is fired, the readout data is actually filled with the masks. In this case we therefore check that the masks are as expected (i.e. they are compatible with the masks that are transmitted at the SOX)
