<!-- doxy
\page refEMCALworkflow EMCAL reconstruction workflow
/doxy -->

# The EMCAL reconstruction workflow

## EMCAL DCS DP processing:

example with test CCDB :

```shell
o2-calibration-emc-dcs-sim-workflow --max-timeframes  10 --delta-fraction 0.5 -b |
o2-calibration-emc-dcs-workflow --use-ccdb-to-configure --use-verbose-mode -b |
o2-calibration-ccdb-populator-workflow --ccdb-path="http://ccdb-test.cern.ch:8080" -b --run
```
For creating EMCAL DCS config in CCDB:
`O2/Detectors/EMCAL/calib/macros/makeEMCALCCDBEntryForDCS.C`

For reading EMCAL DCS data from CCDB:
`O2/Detectors/EMCAL/calib/macros/readEMCALDCSentries.C`
