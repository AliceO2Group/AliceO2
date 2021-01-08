<!-- doxy
\page refMUONMIDWorkflow MID Workflow
/doxy -->

# MID reconstruction workflow

The MID reconstruction starts from the digits and produced MID tracks.
The input digits can be:

- MC digits
- digits obtained after decoding raw data
- digits read from CTF

In the case of the MC digits, the MC labels are propagated as well, thus allowing to relate the reconstructed tracks with the corresponding generated particles.
The procedure to run the reconstruction, either from the digits or from raw data, is detailed in the following.

## Preface: getting the digits

If you do not have the digits, you can obtain a sample with:

```bash
o2-sim -g fwmugen -m MID -n 100
o2-sim-digitizer-workflow
```

## Reconstruction from MC digits

To reconstruct the MC digits, run:

```bash
o2-mid-digits-reader-workflow | o2-mid-reco-workflow
```

### Zero suppression

The MID electronics has a default zero suppression mode. Digits are transmitted only if there is at least one strip fired in both the bending and non-bending plane in at least one of the 4 RPCs which are read-out by a local board.
The zero suppression is not applied to the MC digits that are stored on disk.
This allows to decide whether to apply the zero suppression or not at a later stage, since this mode can be disabled on data.

The digit reader workflow reads the MC digits and applies the zero suppression as default, so that the output is compatible with what would be expected from raw data.
However, one can disable the zero suppression by running the digits reader with:

```bash
o2-mid-digits-reader-workflow --disable-zero-suppression
```

## Reconstruction from raw data

To reconstruct the raw data (either from converted MC digits or real data), run:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-to-digits-workflow | o2-mid-reco-workflow --disable-mc
```

The reconstruction from raw data can also be tested using as input raw data obtained from the MC digits.

### From MC digits to raw data

To convert the MC digits into raw data format, run:

```bash
o2-mid-digits-to-raw-workflow
```

The output will be a binary file named by default *raw_mid.dat*.
Notice that the executable also generates a configuration file that is needed to read the file with the raw reader workflow (see [here](../../../Raw/README.md) for further details)

## From CTF

The CTF for MID corresponds to the digit.
So one can retrieve the digits from the CTF and run the reconstruction with the usual workflow with:

```bash
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MID | o2-mid-reco-workflow --disable-mc
```

### Generate CTF

The MID contribution can be added to CTF by attaching the `o2-mid-entropy-encoder-workflow` device to reconstruction workflow ending by CTF writer, e.g.:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-to-digits-workflow | o2-mid-entropy-encoder-workflow | o2-ctf-writer-workflow
```

## Timing

In each device belonging to the reconstruction workflow, the execution time is measured using the `chrono` c++ library.
At the end of the execution, when the *stop* command is launched, the execution time is written to the `LOG(INFO)`.
An example output is the following:

```less
Processing time / 90 ROFs: full: 3.55542 us  tracking: 2.02182 us
```

Two timing values are provided: one is for the full execution of the device (including retrieval and sending of the DPL messages) and one which concerns only the execution of the algorithm (the tracking algorithm in the above example)
The timing refers to the time needed to process one read-out-frame, i.e. one event.

## Reconstruction options

By default, the reconstruction produces clusters and tracks that are written on file.
It is however possible to only run clustering with:

```bash
o2-mid-reco-workflow --disable-tracking
```

It is also possible to avoid producing a root file with:

```bash
o2-mid-reco-workflow --disable-root-output
```
