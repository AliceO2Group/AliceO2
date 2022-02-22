<!-- doxy
\page refDetectorsTOFcalibration Calibration
/doxy -->

# ccdb object are produced via different workflows

# LHCphase (example with 5 minutes time slots)
# validity -> 5 minutes
o2-calibration-tof-calib-workflow -b --do-lhc-phase --tf-per-slot 26400

# Diagnostics (example with 5 minutes time slots)
# validity -> 5 minutes
o2-calibration-tof-diagnostic-workflow -b --tf-per-slot 26400

# channel offsets
# validity -> from now(1 TF) to infinity
o2-calibration-tof-calib-workflow  --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 100000 -b

# channel offsets per FEA
# validity -> from now(1 TF) to infinity
o2-calibration-tof-calib-workflow  --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 100000 -b --perstrip

# they can be put in the same workflow (for instance reading from file) with populator at the end
o2-tof-calib-reader  --collection-infile listacal --shm-segment-size 5000000000 -b --enable-dia \
| o2-calibration-tof-calib-workflow -b --do-lhc-phase --tf-per-slot 26400 \
| o2-calibration-tof-diagnostic-workflow -b --tf-per-slot 26400 \
| o2-calibration-tof-calib-workflow  --do-channel-offset --update-at-end-of-run-only --min-entries 8 --range 100000 -b \
|o2-calibration-ccdb-populator-workflow -b 

# if you run on epn please add this option to populator to access production ccdb as internal
o2-tof-calib-reader ... \
|o2-calibration-ccdb-populator-workflow -b --ccdb-path "http://o2-ccdb.internal"

# in sync phase reader is replaced by input proxy collecting data from epns
