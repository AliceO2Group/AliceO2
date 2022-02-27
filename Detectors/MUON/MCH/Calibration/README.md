<!-- doxy
\page refDetectorsMUONMCHCalibration Calibration
/doxy -->

# MCH Calibration

## Calibration type(s)

So far MCH has only one calibration : the bad channel map.

This bad channel map is computed from pedestal data taken in special
calibration runs, where the actual zero suppression is disabled in the
electronics, so we readout complete information from the detector and can
compute the mean and sigma of the pedestal values for all the channels.

Once the pedestal means and sigmas are known, we declare some channels bad if
their pedestal mean and/or sigma are outside of some limits (defined in
`BadChannelCalibratorParam`, which is a `ConfigurableParam`, hence settable
from the command-line using the `--configKeyValues` option of the workflow).

## Calibration object(s)

Two objects are produced by the calibration workflow (see below),
distinguishable by the subspec they use :

- subspec 0 : a `std::vector` of `DsChannelId` that should end up in the
  regular CCDB. That object is/will be used for filtering digits during the
  reconstruction/simulation phase
- subspec 1 : a `TObjString` wrapping a CSV containing the same information as
  above, that should end up in the DCS CDB. That object is meant to be used by
  DCS to configure the electronics.

## Calibration workflow

The calibration device executable is named
`o2-calibration-mch-badchannel-calib-workflow` (spec name is
`mch-badchannel-calibrator`). It can be configured using the
[MCHBadChannelCalibratorParam](include/MCHCalibration/BadChannelCalibratorParam.h)
keys.

By default the bad channel calibrator only computes the output object at the
end-of-stream (EOS). To alleviate possible issues with EOS not working
properly, one can set special key options to also compute after a given
fraction of channels have reached enough statistics.

For instance (see also `test/example-mch-pedestal-calibration.sh`) :

```shell
o2-raw-tf-reader-workflow --max-tf 10 --input-data \
  $HOME/alice/data/ped/StfBuilder-CH9R-pedestals-ul-with_gnd-without_HV-20210617 | \
o2-mch-pedestal-decoding-workflow | \
o2-calibration-mch-badchannel-calib-workflow \
  --configKeyValues="MCHBadChannelCalibratorParam.minRequiredNofEntriesPerChannel=100;MCHBadChannelCalibratorParam.minRequiredCalibratedFraction=0.5;MCHBadChannelCalibratorParam.onlyAtEndOfStream=false" | \
o2-calibration-ccdb-populator-workflow \
  --ccdb-path http://localhost:6464 \
  --sspec-min 0 --sspec-max 0 | \
o2-calibration-ccdb-populator-workflow \
  --ccdb-path localhost:8484 --sspec-max 1 --sspec-min 1 --name-extention dcs | \
o2-qc \
  --config json://$HOME/cernbox/o2muon/qc/qc_configs/qc-pedestal.json | \
o2-dpl-run \
  --run -b \ 
> ped-calib.log
```

In that example we instruct the calibrator to compute (and forward to the CCDB
populator) the bad channel map when at least 50% of the channels have more than
100 entries. Note that in this case, and depending of the run duration, several
objects might be uploaded, while in the "normal" case
(`onlyAtEndOfStream=true`) there will be only one (or two, if two populators
are used) object will be uploaded.

