<!-- doxy
\page refDetectorsMUONCommon Common
/doxy -->

# Common code for MID and MCH

For the moment the only commonality stems from the DCS to CCDB part, as
both subsystems only transmit DCS datapoints to CCDB, without any particular
treatment. So we use almost the same source code, but generate separate
executables, e.g. to ease the integration with `dcs-proxy`-based workflows.

Note that MCH transmit both the HV and LV values to CCDB, while MID only
transmit HV values.

## DCS to CCDB

To test the DCS to CCDB route you can use the following 3 parts workflow pipeline :

```shell
o2-calibration-mch-dcs-sim-workflow --max-timeframes 600 --max-cycles-no-full-map 10 -b | \
o2-calibration-mch-dcs-processor-workflow --hv-max-size 0 --hv-max-duration 300 --lv-max-size 0 --lv-max-duration 300 -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:6464" -b
```

```shell
o2-calibration-mid-dcs-sim-workflow --max-timeframes 600 --max-cycles-no-full-map 10 -b | \
o2-calibration-mid-dcs-processor-workflow --hv-max-size 0 --hv-max-duration 300 -b | \
o2-calibration-ccdb-populator-workflow --ccdb-path="http://localhost:6464" -b
```

Note that the only difference (besides the mid vs mch naming) is the set of options of the `processor` device (handling just hv for mid and hv+lv for mch).

- `o2-calibration-[mch|mid]-dcs-sim-worfklow` is just generating fake random MCH or MID DCS data points
- `o2-calibration-[mch|mid]-dcs-processor-workflow` gathers the received data points into a container object
- `o2-calibration-ccdb-populator-workflow` uploads the container object to the CCDB (in this example a local dev ccdb).

 The container object that groups the datapoints is considered ready to be shipped either when the data points span a long enough duration (see the `--xxx-max-duration` option(s) of the `o2-calibration-[mch|mid]-dcs-processor-workflow`) or is big enough (see the `--xxx-max-size`  option(s)).

## MCH DCS Data Points

### HV

The MCH high voltage (HV) system is composed of 188 channels :

- 48 channels for stations 1 and 2 (3 HV channel per quadrant x 16 quadrants)
- 140 channels for stations 3, 4, 5 (1 HV channel per slat x 140 slats)

### LV

The MCH low voltage (LV) system is composed of 328 channels :

- 216 channels (108 x 2 different voltage values) to power up the front-end
  electronics (dualsampas)
- 112 channels to power up the readout crates hosting the solar (readout) cards

## MID DCS Data Points

### HV

The MID high voltage (HV) system is composed of 72 channels, one channel per RPC.

## CCDB quick check

Besides the web browsing of the CCDB, another quick check can be performed with the `o2-[mch|mid]-dcs-ccdb` program to dump the DCS datapoints (hv, lv, or both) or the datapoint config valid at a given timestamp.

```shell
o2-mch-dcs-ccdb --help
o2-mch-dcs-ccdb --ccdb http://localhost:6464 --query hv --query lv --query dpconf
o2-mid-dcs-ccdb --ccdb http://localhost:6464 --query hv --query dpconf
```

The same programs can be used to upload to CCDB the DCS data point configuration for the dcs-proxy :

```shell
o2-mch-dcs-ccdb --put-datapoint-config --ccdb http://localhost:8080
o2-mid-dcs-ccdb --put-datapoint-config --ccdb http://localhost:8080
```

### Default CCDB object

The default object in the CCDB can be produced with:

```shell
o2-mid-dcs-ccdb --ccdb http://localhost:8080 --upload-default-values -t 1662532507890
```

The timestamp represent the last time of validity. This is needed because the default object was added after the CCDB feeding was active and would therefore take over the old measured values.
