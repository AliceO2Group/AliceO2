<!-- doxy
\page refDetectorsMUONMCHConditions Conditions
/doxy -->

# MCH Conditions

The condition data we have are :

- the DCS datapoints (for HV and LV)
- the Bad Channel list (obtained from pedestal calibration runs)
- the Reject list (manual input)

Those objects are stored at the following CCDB paths :

- MCH/Calib/HV
- MCH/Calib/LV
- MCH/Calib/BadChannel
- MCH/Calib/RejectList

The BadChannel and RejectList objects can be uploaded, e.g. for debug purposes, using the `o2-mch-bad-channels-ccdb` program :

```shell
$ o2-mch-bad-channels-ccdb --help
This program dump MCH bad channels CCDB object
Usage:
  -h [ --help ]                         produce help message
  -c [ --ccdb ] arg (=http://localhost:6464)
                                        ccdb url
  --starttimestamp arg (=1677687518645) timestamp for query or put -
                                        (default=now)
  --endtimestamp arg (=1677773918645)   end of validity (for put) - default=1
                                        day from now
  -p [ --put ]                          upload bad channel default object
  -u [ --upload-default-values ]        upload default values
  -t [ --type ] arg (=BadChannel)       type of bad channel (BadChannel or
                                        RejectList)
  -q [ --query ]                        dump bad channel object from CCDB
  -v [ --verbose ]                      verbose output
  -s [ --solar ] arg                    solar ids to reject

```

For instance, to create a debug RejectList object which declares solar number 32 as bad within a local CCDB, from Tuesday 1 November 2022 00:00:01 UTC to Saturday 31 December 2022 23:59:59, use :

```shell
$ o2-mch-bad-channels-ccdb -p -s 32 -t RejectList --starttimestamp 1667260801000 --endtimestamp 1672531199000
storing default MCH bad channels (valid from 1667260801000to 1672531199000) to MCH/Calib/RejectList
```

## Statusmap generation

The status map is an object that list all pads that are not perfect, for some reason. Each such pad gets ascribed a `uint32_t` integer mask, representing the source of information that was used to decide that pad is bad.

The gathering of all the sources of information is performed by the [StatusMapCreatorSpec](src/StatusMapCreatorSpec.cxx), either in standalone [o2-mch-statusmap-creator-workflow](src/statusmap-creator-workflow.cxx) device, or, most probably, as part of the regular `o2-mch-reco-workflow`.

So far only we have only implemented the bad channel list from pedestal runs the manual reject list. Next in line will be the usage of the HV and LV values.

## Statusmap usage

The statusmap, once computed by the statusmap creator, can then be used by the digit filtering, e.g. during reconstruction. To that end some options (by means of the `--configKeyValues` option) must be passed to the statusmap creator and the digit filtering. For instance, to use the information from both the pedestal runs and the rejectlist to compute the statusmap, and to reject pads that are in the resulting statusmap, use :

```c++
MCHStatusMap.useBadChannels=true;MCHStatusMap.useRejectList=true;MCHDigitFilter.statusMask=3
```

where the `statusMask=3` comes from `1 | 2` (see `kBadPedestals | kRejectList` from [StatusMap.h](include/MCHConditions/StatusMap.h])