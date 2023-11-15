<!-- doxy
\page refDetectorsMUONMCHPreClustering Preclustering
/doxy -->

# MCH Preclustering

## Brief description of the algorithm

For each DE, build preclusters on each cathod separately (recursive loop over fired pads and their neighbours), then merge recusively overlapping preclusters between cathods.

## Parameters of the algorithm

Preclustering configurable parameters are defined in [PreClusterFinderParam](include/MCHPreClustering/PreClusterFinderParam.h).

The flag `excludeCorners` (disabled by default) allows to exclude neighbouring pads connected only by corners when looking for fired neighbours.

## Workflow

```shell
o2-mch-digits-to-preclusters-workflow
```

Take as input the list of all digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)) in the current time frame, with the data description "DIGITS", and the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the digits associated to each interaction.
The ROF records input can have the data description "inputrofs:MCH/DIGITROFS" if the direct output of the raw decoder is used, or "inputrofs:MCH/TIMECLUSTERROFS" if the time clustering output is used (default option). The ROF input description can be set on the command line via the `rof-spec` option.

Send the list of all preclusters ([PreCluster](../Base/include/MCHBase/PreCluster.h)) in the time frame, the list of all associated digits ([Digit](/DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/Digit.h)), the list of ROF records ([ROFRecord](../../../../DataFormats/Detectors/MUON/MCH/include/DataFormatsMCH/ROFRecord.h)) pointing to the preclusters associated to each interaction and the list of processing errors ([Error](../Base/include/MCHBase/Error.h)) in four separate messages with the data description "PRECLUSTERS", "PRECLUSTERDIGITS", "PRECLUSTERROFS" and "PRECLUSTERERRORS", respectively.

## Workflow / Spec options

Option `--check-no-leftover-digits xxx` allows to drop an error message (`xxx = "error"` (default)) or an exception (`xxx = "fatal"`) in case not all the input digits end up in a precluster, or to disable this check (`xxx = "off"`).

Option `--sanity-check` allows to perform some input digit sanity checks.

Option `--discard-high-occupancy-des` allows to discard DEs with occupancy > 20%.

Option `--discard-high-occupancy-events` allows to discard events with 5 DEs or more above 20% occupancy.

Option `--configKeyValues "key1=value1;key2=value2;..."` allows to change the preclustering parameters from the command line.

* Example of parameter changed from the command line:
```shell
--configKeyValues "MCHPreClustering.excludeCorners=true"
```
