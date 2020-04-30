<!-- doxy
\page refMUONMIDRaw MID RAW library
/doxy -->

# MID Raw
This directory contains the classes to handle the MID RAW data
This is particularly important for testing and debugging the MID Read-out.

## MID FEEId config
The MID readout consists of 16 crates, divided in two sides.
Each side is read out by one CRU, which contains 24 links divided in two end points.
The readout of one crate requires 2 links, so a total of 16 active GBT links per side is required.

Since each crate has its own logical ID, it is natural to assign to each of the 32 active GBTs a feeId which is defined by the logical crate ID and the ID of the GBT in that crate.
However, each GBT link has its own physical ID, which is given by its cruId, endPointId and linkId.

The class FEEIdConfig, provides an easy way to associate the physical GBT link, with the corresponding feeId.
The configuration can be loaded from a file, which should be in the form:
```
feeId linkId endPointId cruId
```
and should contain 1 line per configured link.
