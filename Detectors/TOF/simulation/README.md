<!-- doxy
\page refDetectorsTOFsimulation Simulation
/doxy -->

# to run digitisation with ccdb object enabling ccdb to anchor to data -> --use-ccdb-tof and --timestamp-tof 1635567000
# second argument will be marked as obsolete once the framework will provide the timestamp
o2-sim-digitizer-workflow -b --onlyDet TOF --use-ccdb-tof --timestamp-tof 1635567000 --shm-segment-size 5000000000

# you can attach also digit QC to check the output
|o2-qc -b --config json://$QUALITYCONTROL_ROOT/etc/tofdigits.json
# for a more detailed output you can replace the data-policy of TaskDigits from tof-digits to tof-digits-dia and then enabling Diagnostic in TaskParameters

# ccdb objects loaded
# diagostic frequency (TRM errors, Readout errors, noisy)
# channel offset+time sleewing (decalibration)
# TOFFEElight map -> active channels
