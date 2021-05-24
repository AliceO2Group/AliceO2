<!-- doxy
\page refDetectorsPHOScalibration PHOS Calibration
/doxy -->

# PHOS calibration objects (see DataDormats/Detectors/PHOS/)

The following objects are used for PHOS calibration
## Pedestals
Calculated during dedicate runs with and used for FEE configuration in later physics runs

## CalibParams
Include channel-by channel gain parameters, High Gain/Low Gain ratio and time offsets for High Gain and Low Gain channels separately.

## TriggerMap
Object contains trigger bad map and 14 parameterizations of L0 turn-on curves (one for each DDL)

## Runbyrun
Time-dependent energy scale corrections, one number per module (plus uncertainty of the fitting procedure)

## BadChannelsMap
Bad channels map used in reconstruction and simulation. Caclulated by combining inputs from pedestal, noisy, LED and physics runs and DCS configuration

# Calibration procedures

## Pedestal calculation
Dedicated pedestal runs (trigger rate ~10 Hz) and length ~100 s to collect ~1000 events. Analyzed with class PHOSPedestalCalibDevice which uses as input vector of o2::phos::Cell, produced with raw to cell converted with option --pedestal. In this conficuration Cell::Energy contains mean and Cell::Time contains RMS of pedestal.
``` cpp
o2-raw-file-reader-workflow --input-conf PHSraw.cfg |
o2-phos-reco-workflow --input-type raw --output-type cells --disable-root-output --pedestal on |
o2-phos-calib-workflow --pedestals
```

# High Gain/Low Gain ratio calculation
Dedicated LED runs (trigger rate ~1000 Hz) and length ~100 s to collect 10^5 events. Analyzed with class PHOSHGLGRatioCalibDevice which uses output of standard reconstruction but wich switched off merging of HG and LG channels:
``` cpp
o2-raw-file-reader-workflow --input-conf PHSraw.cfg |
o2-phos-reco-workflow --input-type raw --output-type cells --disable-root-output --keepHGLG on |
o2-phos-calib-workflow --hglgratio
```
## Turn-on curve and trigger bad map calculation
Run over physics run, selecting events not marked as PHOS L0 trigger. Performed with classes PHOSTurnonCalibDevice and PHOSTurnonCalibrator. Stores transient object to CCDB if sufficient statistics not collected yet or calculates TriggerMap objects and stores it at CCDB. Uses list of Cells and FullClusters as input. Running on MC digits
``` cpp
o2-phos-reco-workflow --input-type digits --output-type cells --disable-root-output |
o2-phos-reco-workflow --input-type cells --output-type clusters --fullclu-output --disable-root-output --disable-root-input |
o2-phos-calib-workflow  --turnon
```

## Run-by-run correction
Calculated pi0 peak position in each module in current run (or equivavlt period). Use list of FullClusters as input and produce 8 numbers: peak position and fit uncertainties for each module. Run as
``` cpp
o2-phos-reco-workflow --input-type digits --output-type cells --disable-root-output |
o2-phos-reco-workflow --input-type cells --output-type clusters --fullclu-output --disable-root-output --disable-root-input |
o2-phos-calib-workflow  --runbyrun
```

## Energy and time calibration
Relative energy scale per channel and time offsets for High Gain and Low Gain channels is calculated with iterative procedure. At the first step physics (better PHOS triggered) data are scanned and list of digits contributed to PHOS clusters are stored in local root files on EPN. digits are packed as std::vector<uint32_t> and simutaneously a list of inv. mass and time histograms are filled. Then new calibration coefficients are calculated from collected histograms and list of clusters re-scanned with new calibration. Few (~5) iterations is necessary to reach final accuracy. First scan is ran as
``` cpp
o2-phos-reco-workflow --input-type digits --output-type cells --disable-root-output |
o2-phos-reco-workflow --input-type cells --output-type clusters --fullclu-output --disable-root-output --disable-root-input |
o2-phos-calib-workflow --not-use-ccdb --energy
```

## Bad map calculation


<!-- doxy
/doxy -->
