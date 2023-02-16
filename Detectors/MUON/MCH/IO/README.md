<!-- doxy
\page refDetectorsMUONMCHIO IO
/doxy -->

# I/O tools for reading and writing MCH data in Root format

<!-- vim-markdown-toc GFM -->

* [Digit reader](#digit-reader)
* [Digit writer](#digit-writer)
* [Precluster reader](#precluster-reader)
* [Precluster writer](#precluster-writer)

<!-- vim-markdown-toc -->

## Digit reader

```shell
o2-mch-digits-reader-workflow
```

For historical reason the digit reader was targeted at simulated digits (for
which we have an extra branch containing MClabel information), so to read real
digit files, you'll unfortunately must use the `--disable-mc` option.

The digit file to be read can be specified with the `--mch-digit-infile` option.

For some expert usage you can also specify the data specifications that will be
used for the digits and the rofs messages (respectively `MCH/DIGITS` and
`MCH/DIGITROFS` by default) using the `--mch-output-digits-data-description` and
`-mch-output-digitrofs-data-description`options.

## Digit writer

```shell
o2-mch-digits-writer-workflow
```

The digit file to be written can be specified with the `--mch-digit-outfile` option.

For some expert usage you can also specify the data specifications that will be
used as sources for the digits and the rofs messages (respectively `MCH/DIGITS` and
`MCH/DIGITROFS` by default) using the `--mch-input-digits-data-description` and
`-mch-input-digitrofs-data-description`options.

## Precluster reader

```shell
o2-mch-preclusters-reader-workflow --infile mchpreclusters.root
```

Send the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, with the data description "PRECLUSTERS", the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERROFS", and the list of digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) associated to preclusters, with the data description "PRECLUSTERDIGITS".

Option `--input-dir` allows to set the name of the directory containing the input file (default = current directory).

Option `--enable-mc` allows to also send the precluster MC labels with the data description "PRECLUSTERLABELS".

## Precluster writer

```shell
o2-mch-preclusters-writer-workflow
```

Take as input the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the current time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction, with the data description "PRECLUSTERS", "PRECLUSTERDIGITS" and "PRECLUSTERROFS", respectively, and write them in the root file "mchpreclusters.root".

Option `--enable-mc` allows to also write the precluster MC labels, with the data description "PRECLUSTERLABELS".

