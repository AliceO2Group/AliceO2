# Configuration processor rollout

`FITDCSConfigProcessor.h` provides the common base for detector processors that
handle dead-channel maps and FEE configurations. Detector implementations supply
their own `init()` and `run()` methods and use the protected setup and parsing
helpers. HV configuration support is deferred.

The FT0, FDD, and FV0 processors currently handle dead-channel maps only and
pass an empty FEE data description to the common base. Detector-specific FEE
readers and outputs can be added in subsequent changes. Dead-channel maps use
`FITDeadChannelMapReader`, formerly named `FITDCSConfigReader`.

# Testing DCS workflow locally

## Local CCDB instance
Local CCDB instance is available here: `http://alimonitor.cern.ch/download/local.jar`
Run it with `java -jar local.jar`.
Configuration is done via environment variables

| Name | Description |
| ---  | --- |
| FILE_REPOSITORY_LOCATION | repository directory, default `/tmp/QC` |
| TOMCAT_ADDRESS | CCDB address, default `127.0.0.1` |
| TOMCAT_PORT | CCDB port, default `8080` |

## Compilation of DCS helpers



## Running DCS workflow


```bash
#!/bin/bash
DET="FDD"
CHANFROM="type=sub,method=connect,address=tcp://127.0.0.1:5556,rateLogging=1,transport=zeromq"
CHANACK="type=push,method=connect,address=tcp://127.0.0.1:5557,rateLogging=1,transport=zeromq"

o2-dcs-config-proxy --subscribe-to "${CHANFROM}" --acknowlege-to "${CHANACK}" \
| o2-fdd-dcs-config-workflow \
| o2-calibration-ccdb-populator-workflow --ccdb-path=http://localhost:18080

```
