<!-- doxy
\page refMUONMIDFilteringExe MID Filtering executable
/doxy -->

# MID filtering

This directory contains executable code to analyse calibration triggers and the subsequent FET event in order to produce masks.

## MID mask maker

This utility allows produce the MID masks
Basic usage:

```bash
o2-mid-mask-maker --feeId-config-file feeId_filename filename [filename_2 filename_3 ...]
```

The `feeId_filename` is a file allowing to tell which feeId is readout by the configured GBT.
The file should be in the form explained [here](../../Raw/README.md)

The executable prints the (possible) list of noisy channels and dead channels and the corresponding masks.
The mask is provided in two formats:

- a digit masks in the form of a ColumnData
- the corresponding masks at the Local Board level. This is useful because it allows to more easily identify the masks to be set in the electronics via DCS.