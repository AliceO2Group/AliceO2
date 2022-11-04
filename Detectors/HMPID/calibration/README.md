#HMPID calibration

For a local test, use

``` o2 - calibration - hmpid - dcs - sim - workflow-- max - timeframes 100 --delta - fraction 0.5 - b |
                    o2 - calibration - hmpid - dcs - workflow - b-- local - test | o2 - calibration - ccdb - populator - workflow-- ccdb - path localhost : 8080 - b-- run

```

`time - frames` specifies the timeframes before EOR
`local - test` specifies to use simulated DPs
`ccdb - path localhost : 8080` specifies to store CCDB objects locally
