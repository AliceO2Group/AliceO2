<!-- doxy
\page refITScalibration testWorkflow
/doxy -->

# ITS calibration workflows

## Generation of fixed-pattern noise maps out of pass1 clusters:

This will read clusters from a ROOT file and write a noise map to a local CCDB running on port 8080:

```shell
o2-its-cluster-reader-workflow -b | \
o2-calibration-its-calib-workflow -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" -b
```
The same as above, but starting from raw data, could be done like this:

```shell
o2-raw-file-reader-workflow --detect-tf0 --input-conf run.cfg --delay 0.1 | \
o2-itsmft-stf-decoder-workflow -b --nthreads 4 --shm-segment-size 16000000000 | \
o2-calibration-its-calib-workflow -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:8080" -b
```
