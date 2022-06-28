<!-- doxy
\page refDetectorsMUONMCHTimeClustering Time Clustering
/doxy -->

# MCH Time Clustering

Regroups (aka merges) ROFs which have compatible times, making ROFs that span larger durations.

```shell
o2-mch-digits-to-timeclusters-workflow
```

Take as input the list of all digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the data description "DIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each interaction, with the data description "DIGITROFS". Send a new list of ROF records that combine all the digits correlated in time within a user-defined time window, with the data description "TIMECLUSTERROFS".

The time clustering can be configured using `--configKeyValues` :

```shell
o2-mch-digits-to-timeclusters-workflow --configKeyValues="MCHTimeClustering.maxClusterWidth=xx;MCHTimeClustering.peakSearchNbins=yy"
```

`maxClusterWidth` allows to set the width of the time correlation window.
The time clustering is based on a brute-force peak search algorithm, which arranges the input digits into coarse time bins. The number of bins in one time cluster window can be set via the `peakSearchNbins` parameter.

