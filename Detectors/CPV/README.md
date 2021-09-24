<!-- doxy
\page refDetectorsCPV CPV
/doxy -->

# CPV 

CPV stands for Charged Particles Veto detector which actually is pad chamber with cathode pad readout.
There are 3 CPV modules. Each module seats on top of PHOS modules M2, M3 and M4, so the numeration of CPV modules is naturally same: M2, M3, M4.
Each module has 128 x 60 = 7680 channels.
It is triggered detector and is using CRU (Common Readout Unit) for readout in LHC Run3. 
More details can be found [here](https://twiki.cern.ch/twiki/bin/viewauth/ALICE/CPV).

## Readout
CPV readout is organized in 3 GBT links, all connected to single CRU card on single FLP. 
Triggered events are packed within each HeartBeatFrame. More info about data format can be found [here](https://twiki.cern.ch/twiki/pub/ALICE/CPV/cpv_data_format.pdf).

## Reconstruction
The reconstruction is steered via [o2-cpv-reco-workflow](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/workflow/src/cpv-reco-workflow.cxx) executable.
#### First,
it starts directly on FLP. Raw data is provided to [RawToDigitConverter](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/workflow/src/RawToDigitConverterSpec.cxx) 
which converts raw format to cpv [digits](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Digit.h) and [trigger records](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/TriggerRecord.h).
Digits are calibrated objects: RawToDigitConverter reads 
[pedestals](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Pedestals.h), 
[bad channel map](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/BadChannelMap.h) 
and [gain](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/CalibParams.h) 
calibration objects from CCDB, then it excludes bad channels, subtracts pedestals from raw amplitudes,
and result is multiplied by gain calibration coefficient forming the digit which keeps calibrated signal and its channel address.
Trigger records are objects which are keeping a reference to digits belonging to same event. To start conversion of raw data to digits run following command:
```shell
o2-cpv-reco-workflow --input-type raw --output-type digits --disable-mc --disable-root-output
```
If there is no need to calibrate digits, add `--pedestal` flag to `o2-cpv-reco-workflow` then digit signal will be equal to raw amplitude. Such mode is used for pedestal calibration (see below).
If you want to redirect stream of digits to root file then remove `--disable-root-output`.
If you want to process MC and you need digits to be labeled according to corresponding primary particles then remove `--disable-mc` flag.

#### Second, 
output of previous command is stream of digits and triggers records. It's expected that the stream from FLP goes to EPNs where clusterization procedure is expected to run.
Clusterization is also steered by `o2-cpv-reco-workflow` executable. In order to run clusterization on digits, run 
```shell
o2-cpv-reco-workflow --input-type digits --output-type clusters --disable-mc --disable-root-output
```
The output of the command is stream of [clusters](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Cluster.h) and corresponding trigger records.
Then clusters are ready to be compressed to CTF and to be keeped on storage. 

## Simulation

## Calibration

## Quality Conrol

## Local testing
