date

o2-raw-tf-reader-workflow --input-data \
    $HOME/alice/data/2022/LHC22t_MCH/529437/calib/1140 --onlyDet MCH --max-tf 1000 | \
o2-mch-pedestal-decoding-workflow | \
o2-calibration-mch-badchannel-calib-workflow \
  --configKeyValues="MCHBadChannelCalibratorParam.minRequiredNofEntriesPerChannel=10;MCHBadChannelCalibratorParam.minRequiredCalibratedFraction=0.5;MCHBadChannelCalibratorParam.onlyAtEndOfStream=true" | \
o2-calibration-ccdb-populator-workflow \
    --ccdb-path http://localhost:6464 \
    --sspec-min 0 --sspec-max 0 | \
o2-calibration-ccdb-populator-workflow \
    --ccdb-path localhost:8484 --sspec-max 1 --sspec-min 1 --name-extention dcs | \
o2-dpl-run --run -b

date
