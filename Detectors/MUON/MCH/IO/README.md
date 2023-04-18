<!-- doxy
\page refDetectorsMUONMCHIO IO
/doxy -->

# I/O tools for reading and writing MCH data in Root format

<!-- vim-markdown-toc GFM -->

* [Digit reader](#digit-reader)
* [Digit writer](#digit-writer)
* [Precluster reader](#precluster-reader)
* [Precluster writer](#precluster-writer)
* [Cluster reader](#cluster-reader)
* [Cluster writer](#cluster-writer)
* [Track reader](#track-reader)
* [Track writer](#track-writer)

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

## Cluster reader

```shell
o2-mch-clusters-reader-workflow --infile mchclusters.root [--enable-mc] [--local] [--no-digits]
```

Send the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "GLOBALCLUSTERS", the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS", and the list of digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) associated to clusters, with the data description "CLUSTERDIGITS".

Option `--local` assumes that clusters are in the local coordinate system and send them with the description "CLUSTERS".

Option `--no-digits` allows to do not send the associated digits.

Option `--enable-mc` allows to also send the cluster MC labels with the data description "CLUSTERLABELS".

## Cluster writer

```shell
o2-mch-clusters-writer-workflow [--enable-mc] [--local] [--no-digits]
```

Take as input the list of all clusters ([Cluster](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Cluster.h)) in the current time frame, with the data description "GLOBALCLUSTERS", the list of associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)), with the data description "CLUSTERDIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the clusters associated to each interaction, with the data description "CLUSTERROFS", and write them in the root file "mchclusters.root".

Option `--local` allows to write the list of clusters in the local coordinate system, with the data description "CLUSTERS".

Option `--no-digits` allows to do not write the associated digits.

Option `--enable-mc` allows to also write the cluster MC labels, with the data description "CLUSTERLABELS".

## Track reader

```shell
o2-mch-tracks-reader-workflow --infile mchtracks.root
```

Does the same work as the [Track sampler](#track-sampler) but starting from a Root file (`mchtracks.root`)  containing `TRACKS`, `TRACKROFS` and `TRACKCLUSTERS` containers written e.g. by the [o2-mch-tracks-writer-workflow](#track-writer).
Note that a very basic utility also exists to get a textual dump of a Root tracks file : `o2-mch-tracks-file-dumper`.

Option `--input-dir` allows to set the name of the directory containing the input file (default = current directory).

Option `--digits` allows to also read the associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) and send them with the data description "TRACKDIGITS".

Option `--enable-mc` allows to also read the track MC labels and send them with the data description "TRACKLABELS".

Option `--subspec` allows to specify a subspec (default is 0) so that several `mchtracks.root` files can be read in the same workflow (usefull for instance in a comparison workflow).

## Track writer

```shell
o2-mch-tracks-writer-workflow --outfile "mchtracks.root"
```

Does the same kind of work as the [track sink](#track-sink) but the output is in Root format instead of custom binary one. It is implemented using the generic [MakeRootTreeWriterSpec](/DPLUtils/MakeRootTreeWriterSpec.h) and thus offers the same options.

Option `--digits` allows to also write the associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) from the input message with the data description "TRACKDIGITS".

Option `--enable-mc` allows to also write the track MC labels from the input message with the data description "TRACKLABELS".

