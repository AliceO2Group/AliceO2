Local example workflow with local CCDB (running on port 6464) :

```shell
o2-dcs-random-data-workflow --max-timeframes=10 | 
o2-calibration-ccdb-populator-workflow --ccdb-path http://localhost:6464
```
