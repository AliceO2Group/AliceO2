<!-- doxy
\page refDetectorsTOFcalibration Calibration
/doxy -->

FOR ASYNC TOF CALIBRATION

# set ccdb url (please check!)
export ccdburl="ccdb-test.cern.ch:8080"
export runNum="555555"

# calibrate diagnostic
o2-tof-calib-reader -b  --collection-infile listacal --configKeyValues "HBFUtils.runNumber=$runNum" \
| o2-calibration-tof-diagnostic-workflow -b --tf-per-slot 26400 --max-delay 0 --condition-tf-per-query -1 \
|o2-calibration-ccdb-populator-workflow -b --ccdb-path $ccdburl

# calibrate LHC phase
o2-tof-calib-reader -b  --collection-infile listacal --configKeyValues "HBFUtils.runNumber=$runNum" \
| o2-calibration-tof-calib-workflow -b --do-lhc-phase --tf-per-slot 26400 --max-delay 0 --condition-tf-per-query -1 --use-ccdb \
|o2-calibration-ccdb-populator-workflow -b --ccdb-path $ccdburl

# calibrate offset
# check after 100 s if ccdb is updated (for LHCphase)
o2-tof-calib-reader -b  --collection-infile listacal --configKeyValues "HBFUtils.runNumber=$runNum" \
| o2-calibration-tof-calib-workflow  --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 100000 -b --condition-tf-per-query 8800 --use-ccdb \
| o2-calibration-ccdb-populator-workflow -b --ccdb-path $ccdburl

# calibrate Time Slewing -> for the moment just re-run offset with refined time range (assuming better channel alignment)
o2-tof-calib-reader -b  --collection-infile listacal --configKeyValues "HBFUtils.runNumber=$runNum" \
| o2-calibration-tof-calib-workflow  --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 10000 -b --condition-tf-per-query 8800 --use-ccdb \
| o2-calibration-ccdb-populator-workflow -b --ccdb-path $ccdburl

# if you run on epn please add this option to populator to access production ccdb as internal
o2-tof-calib-reader ... \
|o2-calibration-ccdb-populator-workflow -b --ccdb-path "http://o2-ccdb.internal"

# in sync phase reader is replaced by input proxy collecting data from epns
