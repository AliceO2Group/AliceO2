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

```less
feeId linkId endPointId cruId
```

and should contain 1 line per configured link.

## MID ROBoard configuration

The local and regional board have configuration registers that can be modified to change parameters such as the delays, the masks, and the way the zero suppression is performed.

The delays are not supposed to change, but the masks and the zero suppression can be configurable.
The configuration file is a txt file that is read by WinCC and applied to the electronics via GBT, using the ALF+FRED software.

The list of noisy channels and the corresponding masks are obtained running a dedicate workflow (see [here](../Workflow/README.md) for details).
It is therefore natural for such a workflow to generate a configuration file that can then be passed to DCS.

This file is in the form:

```less
localBoardId MID_LOC_config MID_LOC_maskX1Y1 MID_LOC_maskX2Y2 MID_LOC_maskX3Y3 MID_LOC_maskX4Y4
```

where the MID_LOC* variables are the ones described in the `Detector Control` link [here](http://www-subatech.in2p3.fr/~electro/projets/alice/dimuon/trigger/upgrade/index.html).

These values are coded into the class [ROBoardConfig](include/MIDRaw/ROBoardConfig.h).
The class [ROBoardConfigHandler](include/MIDRaw/ROBoardConfigHandler.h) was created to handle the loading/writing of the parameters from/to the txt file.
