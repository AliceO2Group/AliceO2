<!-- doxy
\page refDetectorsMUONMCHConditions Conditions
/doxy -->

# MCH Conditions

## From DCS

To test the DCS to CCDB route you can use the following 3 parts worfklow pipeline : 

```shell
o2-calibration-mch-dcs-sim-workflow --max-timeframes 10000 --max-cycles-no-full-map 100 -b | \
o2-calibration-mch-dcs-processor-workflow -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:6464" -b
```

- `o2-calibration-mch-dcs-sim-worfklow` is just generating fake random MCH DCS data points, 
- `o2-calibration-mch-dcs-processor-workflow` gathers the received data points into a container object 
- `o2-calibration-ccdb-populator-workflow` uploads the container object to the CCDB (in this example a local dev ccdb).

 The container object that groups the datapoints is considered ready to be shipped either when the data points span a long enough duration (see the `--calib-object-max-duration` option of the `o2-calibration-mch-dcs-processor-workflow`) or is big enough (see the `--calib-object-max-size` option).

### HV

The MCH high voltage (HV) system is composed of 188 channels :

- 48 channels for stations 1 and 2 (3 HV channel per quadrant x 16 quadrants)
- 140 channels for stations 3, 4, 5 (1 HV channel per slat x 140 slats)

### LV

The MCH low voltage (LV) system is composed of 328 channels :

- 216 channels (108 x 2 different voltage values) to power up the front-end
  electronics (dualsampas)
- 112 channels to power up the readout crates hosting the solar (readout) cards


