<!-- doxy
\page refDetectorsCalibration/testMacros Module 'Detectors/Calibration/testMacros'
/doxy -->

# Simulation of sending of calibration data from EPNs to an aggregator

To be used when calibrations produced by several EPNs have to be sent to a single node, the aggregator.
On the aggregator, the devices producing calibration will run and send the output to the CCDB.

* In order to populate the CCDB, the CCDB local server should be started, if the exercise is not meant to
upload the official or test CCDB. In a terminal, run:
```cpp
java -jar local.jar
```
which will start a CCDB server on port 8080. See [instructions](https://github.com/AliceO2Group/AliceO2/tree/dev/CCDB#central-and-local-instances-of-the-ccdb).

* To run the calibration and aggregator, open a terminal and start the aggregator, which is just the `o2-dpl-raw-proxy`:
```cpp
o2-dpl-raw-proxy --dataspec A:TOF/CALIBDATA/0 --channel-config "name=readout-proxy,type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq"
```
where, as you can see, you can overwrite the configurations of the `readout-proxy` channel, which is the channel used by the aggregator.
The aggregator should be started **before the other devices**, since it will listen for data (it is the channel that is *binding*).

* The following example will pass the data arriving to the aggregator to the `o2-calibration-lhc-clockphase-workflow` device, and from there
to the `o2-calibration-ccdb-populator-workflow` to update the CCDB. For this, the raw-proxy should be in pipeline with these workflows like this:
```cpp
o2-dpl-raw-proxy --dataspec A:TOF/CALIBDATA/0 --channel-config "type=pull,method=bind,address=tcp://localhost:30453,rateLogging=1,transport=zeromq,name=readout-proxy" | o2-calibration-lhc-clockphase-workflow --tf-per-slot 20 -b | o2-calibration-ccdb-populator-workflow --ccdb-path localhost:8080
```

* To send data to the aggregator simulating an arbitrary number of EPNs, in another terminal, run:
```cpp
source runEPNsimulation.sh 3
```
where the argument (`3` above) is the number of EPNs to be simulated. This will send the data to the aggregator process. 

**N.B.**: the aggregator and calibration devices will need to use a different port from localHost:8080 in case the local
CCDB server is used in its default configuration, since 8080 is used by CCDB. In the example above, port 30453 is used.
