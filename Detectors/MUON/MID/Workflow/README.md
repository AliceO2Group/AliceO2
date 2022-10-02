<!-- doxy
\page refMUONMIDWorkflow MID Workflow
/doxy -->

# MID workflows

1. [MID reconstruction workflow](#mid-reconstruction-workflow)
2. [MID raw data checker](#mid-raw-data-checker)
3. [MID calibration](#mid-calibration)
4. [MID digits writer](#mid-digits-writer)
5. [MID raw data dumper](#mid-raw-data-dumper)
6. [MID efficiency](#mid-efficiency)

## MID reconstruction workflow

The MID reconstruction starts from the digits and produced MID tracks.
The input digits can be:

- MC digits
- digits obtained after decoding raw data
- digits read from CTF

In the case of the MC digits, the MC labels are propagated as well, thus allowing to relate the reconstructed tracks with the corresponding generated particles.
The procedure to run the reconstruction, either from the digits or from raw data, is detailed in the following.

### Preface: getting the digits

If you do not have the digits, you can obtain a sample with:

```bash
o2-sim -g fwmugen -m MID -n 100
o2-sim-digitizer-workflow
```

### Reconstruction from MC digits

To reconstruct the MC digits, run:

```bash
o2-mid-digits-reader-workflow | o2-mid-reco-workflow
```

#### Zero suppression

The MID electronics has a default zero suppression mode. Digits are transmitted only if there is at least one strip fired in both the bending and non-bending plane in at least one of the 4 RPCs which are read-out by a local board.
The zero suppression is not applied to the MC digits that are stored on disk.
This allows to decide whether to apply the zero suppression or not at a later stage, since this mode can be disabled on data.

The digit reader workflow reads the MC digits and applies the zero suppression as default, so that the output is compatible with what would be expected from raw data.
However, one can disable the zero suppression by running the digits reader with:

```bash
o2-mid-digits-reader-workflow --disable-zero-suppression
```

### Reconstruction from raw data

To reconstruct the raw data (either from converted MC digits or real data), run:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-to-digits-workflow | o2-mid-reco-workflow --disable-mc
```

The reconstruction from raw data can also be tested using as input raw data obtained from the MC digits.

#### From MC digits to raw data

To convert the MC digits into raw data format, run:

```bash
o2-mid-digits-to-raw-workflow
```

The output will be a binary file named by default *raw_mid.dat*.
Notice that the executable also generates a configuration file that is needed to read the file with the raw reader workflow (see [here](../../../Raw/README.md) for further details)

### From CTF

The CTF for MID corresponds to the digit.
So one can retrieve the digits from the CTF and run the reconstruction with the usual workflow with:

```bash
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MID | o2-mid-reco-workflow --disable-mc
```

#### Generate CTF

The MID contribution can be added to CTF by attaching the `o2-mid-entropy-encoder-workflow` device to reconstruction workflow ending by CTF writer, e.g.:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-to-digits-workflow | o2-mid-entropy-encoder-workflow | o2-ctf-writer-workflow
```

### CPU timing

In each device belonging to the reconstruction workflow, the execution time is measured using the `chrono` c++ library.
At the end of the execution, when the *stop* command is launched, the execution time is written to the `LOG(info)`.
An example output is the following:

```less
Processing time / 90 ROFs: full: 3.55542 us  tracking: 2.02182 us
```

Two timing values are provided: one is for the full execution of the device (including retrieval and sending of the DPL messages) and one which concerns only the execution of the algorithm (the tracking algorithm in the above example)
The timing refers to the time needed to process one read-out-frame, i.e. one event.

### Afterburner

There is an offset between the collision BC and the BC that can be obtained from the electronics clock.
This offset is in principle accounted for when decoding the raw data.
However, the precise value of this offset depends on the delays that chosen electronics delay, and some adjustment might be needed.
To avoid having to regenerate the CTF, the time offset of the digits can be adjusted on-the-fly by running the reconstruction with the option:

```bash
o2-mid-reco-workflow --change-local-to-BC <value>
```

where `<value>` is the chosen offset in number of BCs (can be negative).

### Reconstruction options

By default, the reconstruction produces clusters and tracks that are written on file.
It is however possible to only run clustering with:

```bash
o2-mid-reco-workflow --disable-tracking
```

It is also possible to avoid producing a root file with:

```bash
o2-mid-reco-workflow --disable-root-output
```

## MID raw data checker

This workflow is used to check the consistency of the raw data produced from the CRU.
The input is provided by either reading a raw file with the file reader workflow or by using the DPL-proxy.
The common usage is:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-checker-workflow --feeId-config-file "feeId_filename"
```

The `feeId_filename` is a file allowing to tell which feeId is readout by the configured GBT.
The file should be in the form explained [here](../Raw/README.md)/

The workflow generates an output file where one can find:

- the details of the event where the data from one GBT lnk is not consistent (with the reason of the inconsistency)
- a summary of the number of analysed events and the number of events with errors

The default output file name is `raw_checker_out.txt`, but it can be changed with the option: `--mid-checker-outfile`

The decoding and checking of raw data produced without the user logic is time consuming.
In order to be able to speed-up the process, the check can be launch per gbt link.
This is achieved by adding the option: `--per-gbt`.
In this case, the workflow will produce one output per link, which is called: `raw_checker_out_GBT_LINKID.txt`, where `LINKID` is the link number.

## MID calibration

This workflow checks the fired strips in calibration events (when only noisy strips are fired) and during FET events (where all strips alive should fired).
Scalers are filled for the noisy and dead channels, respectively.
The workflow then compares the number of times a noisy channel was fired (or the number of times a channel was dead) to the total number of calibration triggers analysed.
If the fraction is larger than a configurable threshold, a mask is produced to accounts for noisy and dead channels.

The common usage is:

```bash
o2-raw-file-reader-workflow --input-conf MIDraw.cfg | o2-mid-raw-to-digits-workflow | o2-mid-calibration-workflow
```

The fraction of time a strip must be noisy or dead in order to be masked can be adjusted with: `--mid-mask-threshold XX` (with 0<`XX`<= 1).
The scalers are reset from time to time in order to better check the evolution of the noisy/dead channels.

## MID digits writer

This workflow writes to file the decoded digits.
It is useful for debugging.

Usage:

```bash
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MID | o2-mid-decoded-digits-writer-workflow
```

# MID raw data dumper

This workflow allows to dump on screen the raw data.
It is useful for debugging
Usage:

```bash
o2-raw-tf-reader-workflow --onlyDet MID --input-data o2_rawtf_run00505645_tf00000001_epn156.tf --max-tf 1 | o2-mid-raw-dump-workflow
```

If option `--decode` is added, the decoded digits are dumped instead.

# MID efficiency

This workflow allows to compute the MID chamber efficiency.
This is just an example since, eventually, the workflow should be rewritten in order to be able to run on AODs.

Usage:

```bash
o2-ctf-reader-workflow --ctf-input o2_ctf_0000000000.root --onlyDet MID | o2-mid-reco-workflow --disable-mc | o2-mid-efficiency-workflow
```
