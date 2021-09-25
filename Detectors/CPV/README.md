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
#### 1. Raw to digits
It starts directly on FLP. Raw data is provided to [RawToDigitConverter](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/workflow/src/RawToDigitConverterSpec.cxx)
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
Raw decoding itself is done by [RawDecoder](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/reconstruction/include/CPVReconstruction/RawDecoder.h) class.

#### 2. Digits to clusters
Output of previous command is stream of digits and triggers records. It's expected that the stream from FLP goes to EPNs where clusterization procedure is expected to run.
Clusterization is also steered by `o2-cpv-reco-workflow` executable. In order to run clusterization on digits, run
```shell
o2-cpv-reco-workflow --input-type digits --output-type clusters --disable-mc --disable-root-output
```
The output of the command is stream of [clusters](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Cluster.h) and corresponding trigger records. Clusterization is done by [Clusterer](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/reconstruction/include/CPVReconstruction/Clusterer.h) class.

#### 3. Clusters to CTF
Then clusters are ready to be compressed to Compressed Time Frame and to be kept at storage. Try to convert
<!-- add here info how to run CTF creator -->


## Simulation
Simulation is organized as follows:
#### 1. Creation of hits
[Hits](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Hit.h) are objects which keep information about signals such as energy depositions created by single primary particles in detector channels. Hit creation is done by [Detector](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/simulation/include/CPVSimulation/Detector.h) class. To run hits creation type
```shell
 o2-sim -n10 -g boxgen --configKeyValues 'BoxGun.pdg=11 ; BoxGun.prange[0]=10.; BoxGun.prange[1]=15.; BoxGun.phirange[0]=260; BoxGun.phirange[1]=280; BoxGun.number=50; BoxGun.eta[0]=-0.125 ; BoxGun.eta[1]=0.125; ' -m CPV
```
This command will generate 10 events (`-n 10` option) with uniform particle generator (`-g boxgen` option) which generates 50 electrons (`BoxGun.number=50; BoxGun.pdg=11`) with flat momentum distribution in range from 10 GeV/c to 15 GeV/c (`BoxGun.prange[0]=10.; BoxGun.prange[1]=15.;`), flat azimuthal angle phi distribution in range from 260 to 280 degrees (`BoxGun.phirange[0]=260; BoxGun.phirange[1]=280;`), flat pseudorapidity distribution in range from -0.125 to 0.125 units (`BoxGun.eta[0]=-0.125 ; BoxGun.eta[1]=0.125;`), and finally hist only for CPV are created (`-m CPV`). Result of this command is file o2sim_HitsCPV.root containing hits and some other important stuff.

#### 2. Hits to digits
Hits from different primaries then needed to be merged and electronic noise to be added in order to obtain  digits from hits. This can be done with [Digitizer](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/simulation/include/CPVSimulation/Digitizer.h) workflow. Run
```shell
o2-sim-digitizer-workflow --onlyDet CPV
```
in order to do so. It merges hits and adds electronic noise to merged signals. Electronic noise is simulated as 3 sigma pedestal jitter.  As a result the file
You can add `--configKeyValues 'CPVSimParams.mCCDBPath=localtest'` option in order to avoid connection to CCDB and use ideal pedestals (sigma = 1.5 ADC counts for all channels). You can also choose how much sigmas to use for noise simulation providing `--configKeyValues 'CPVSimParams.mZSnSigmas=X'` option, where X is floating point number.

#### 3. Digits to raw
Digits can be converted to raw data format using [RawWriter](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/simulation/include/CPVSimulation/RawWriter.h) class. Try to run
```shell
o2-cpv-digi2raw -o raw/CPV
```
This command will read digits from cpvdigits.root by default and produce raw/CPV/ folder with config and raw files. Digits to raw conversion is done by [RawWriter](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/simulation/include/CPVSimulation/RawWriter.h) class. You can specify `--ccdb-url localtest` option in order to use dummy calibration. IMPORTANT is to use same CCDB path as you used at previous step i.e. hits to digits conversion. Input file also can be changed by providing `-i path/to/file.root` option.


Raw file is ready to be read as normal raw data file and be processed normally with full reconstruction chain like that:
```shell
o2-raw-file-reader-workflow --input-conf raw/CPV/CPVraw.cfg |  o2-cpv-reco-workflow --input-type raw --output-type digits --disable-mc --disable-root-output |  o2-cpv-reco-workflow --input-type digits --output-type clusters --disable-mc
```

## Calibration
Calibration is based on [TimeSlotCalibration](https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/Calibration#readme) framework. It supposed to produce calibration objects valid for certain time intervals and put them into CCDB. Calibration processes are expected to be running on EPN using digits and clusters.

#### Pedestals
Pedestal calibration is needed to measure pedestal values and their RMSs (sigmas). Pedestal value must be subtracted from amplitude at reconstruction stage. Also pedestals values and sigmas are used to configure FEE thresholds (so-called zero suppression) in physics runs. To measure them pedestal run without zero supression must be taken. Then its raw data converted to digits with `--pedestal` flag in order not to calibrate it. Thus digits have signal equal to raw amplitude. Then digit stream is picked up by `o2-calibration-cpv-calib-workflow`. The actual calibration is done by [PedestalCalibrator](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/calib/include/CPVCalibration/PedestalCalibrator.h) class inherited from [TimeSlotCalibration](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/Calibration/include/DetectorsCalibration/TimeSlotCalibration.h). It produces PedestalSpectrum for each time slot and then finalizing it at the end of TimeSlot or at the end of run. Calibration itself can be run with followng command (it must be within some workflow, of cause):
```shell
o2-calibration-cpv-calib-workflow --pedestals --max-delay 0 --tf-per-slot 100
```
Option `--tf-per-slot 100` indicates length of time intervals in TimeFrames. In this particular case length of TimeSlot is 100 TFs. Please consult [this page]() for explanation of all available options. One can also use `--updateAtTheEndOfRunOnly` option in order to finalize TimeSlots and produce calibration object at the end-of-run only. Be careful with that as end-of-run can be unclear yet within the O2 project so it can never happen. At least with the `o2-raw-file-reader-workflow` the finalization of TimeSlot never started yet at the end of file reading. Output of `o2-calibration-cpv-calib-workflow --pedestals` is stream of [Pedestals](https://github.com/AliceO2Group/AliceO2/blob/dev/DataFormats/Detectors/CPV/include/DataFormatsCPV/Pedestals.h) objects and corresponding metadata for CCDB entry. Use `o2-ccdb-populator-workflow` in order to put created objects to CCDB. The overall calibration chain should look like that:
```shell
o2-raw-file-reader-workflow --input-conf CPVraw.cfg  | o2-cpv-reco-workflow --input-type raw --output-type digits --disable-mc --disable-root-output --pedestal | o2-calibration-cpv-calib-workflow --pedestals --max-delay 0 --tf-per-slot 100 | o2-calibration-ccdb-populator-workflow
```
Explanation: it reads file with pedestal data; then raw data is converted to digits without calibration; then digits are picked up by calibrator; and finally produced calibration objects are putted to CCDB. After that it is possible to read them from CCDB with [this script](https://github.com/AliceO2Group/AliceO2/blob/dev/Detectors/CPV/calib/macros/readPedestalsFromCCDB.C).

## Local testing (quick summary)
#### Simulation
```shell
o2-sim -n10 -g boxgen --configKeyValues 'BoxGun.pdg=11 ; BoxGun.prange[0]=10.; BoxGun.prange[1]=15.; BoxGun.phirange[0]=260; BoxGun.phirange[1]=280; BoxGun.number=50; BoxGun.eta[0]=-0.125 ; BoxGun.eta[1]=0.125; ' -m CPV
o2-sim-digitizer-workflow   --onlyDet CPV #consider to add --configKeyValues 'CPVSimParams.mCCDBPath=localtest' to use ideal pedestals for noise simulation
o2-cpv-digi2raw -o raw/CPV
```
#### Reconstruction
```shell
o2-raw-file-reader-workflow --input-conf raw/CPV/CPVraw.cfg |  o2-cpv-reco-workflow --input-type raw --output-type digits --disable-mc --disable-root-output |  o2-cpv-reco-workflow --input-type digits --output-type clusters --disable-mc
```
#### Pedestal calibration
```shell
o2-raw-file-reader-workflow --input-conf CPVraw.cfg  | o2-cpv-reco-workflow --input-type raw --output-type digits --disable-mc --disable-root-output --pedestal | o2-calibration-cpv-calib-workflow --pedestals --max-delay 0 --tf-per-slot 100 | o2-calibration-ccdb-populator-workflow
```
