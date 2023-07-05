# MCH StatusMap

The status map is an object that list all pads that are not perfect, for some reason. Each such pad gets ascribed a `uint32_t` integer mask, representing the source of information that was used to decide that pad is bad.

## StatusMap generation

The gathering of all the sources of information is performed by the [StatusMapCreatorSpec](src/StatusMapCreatorSpec.cxx), either in standalone [o2-mch-statusmap-creator-workflow](src/statusmap-creator-workflow.cxx) device, or, most probably, as part of the regular [o2-mch-reco-workflow](../Workflow/src/reco-workflow.cxx).

So far only we have only implemented the bad channel list from pedestal runs and the manual reject list. Next in line will be the usage of the HV and LV values.

## StatusMap usage

The status map, once computed by the statusmap creator, can then be used by the digit filtering, e.g. during reconstruction. To that end some options (by means of the `--configKeyValues` option) must be passed to the statusmap creator and the digit filtering. For instance, to use the information from both the pedestal runs and the rejectlist to compute the statusmap, and to reject pads that are in the resulting statusmap, use :

```c++
MCHStatusMap.useBadChannels=true;MCHStatusMap.useRejectList=true;MCHDigitFilter.statusMask=3
```

where the `statusMask=3` comes from `1 | 2` (see `kBadPedestals | kRejectList` from [StatusMap.h](include/MCHConditions/StatusMap.h])
