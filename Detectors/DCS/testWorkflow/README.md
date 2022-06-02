<!-- doxy
\page refDetectorsDCStestWorkflow testWorkflow
/doxy -->

# Standalone example

Local example workflow with local CCDB (running on port 6464) :

```shell
o2-dcs-random-data-workflow --max-timeframes=10 |
o2-calibration-ccdb-populator-workflow --ccdb-path http://localhost:6464
```

# Simulation of detector specific data points

In order to test the processing of their datapoints, subsystems can, for instance, setup a basic workflow chain consisting of a simulator, a processor and a ccdb populator.

```console
det-dcs-simulator | det-processor | o2-calibration-ccdb-populator-workflow
```

The simulator must create a message containing a vector of DataPointCompositeObject for the detector. The processor then does "something" with those data points, and creates a set of object pairs (clbInfo,clbPayload) that are transmitted to the ccdb populator to be uploaded to the CCDB.

The ccdb populator is an existing workflow that can be reused by any susbsystem. The processor is of course detector specific and must be written accordingly.
The simulator is also detector specific in the sense that each detector has a different set of datapoints to be simulated. It can be written from scratch if so desired, but it can also be written with the help of `getDCSRandomGeneratorSpec` function (defined in the `Detectors/DCS/testWorkflow/include/DCStestWorkflow/DCSRandomDataGeneratorSpec.h` include file) for cases where random generation of data points is sufficient.

It then boils down to :

```
#include "DCStestWorkflow/DCSRandomDataGeneratorSpec.h"
#include "Framework/runDataProcessing.h"

o2::framework::WorkflowSpec defineDataProcessing(o2::framework::ConfigContext const& configcontext)
{
  std::vector<o2::dcs::test::HintType> dphints;

  // populate the dphints vector with compact description of what
  // data points should be generated using DataPointHint<T>
  //
  // a DataPointHint is a (pattern,min,max) triplet where the pattern
  // is a string pattern that gets exanded to one or several actual
  // DCS alias name(s), and the min and max are the actual range of
  // the values to be generated.

  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_vp_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<double>{"tof_hv_ip_[00..02]", 0, 50.});
  dphints.emplace_back(o2::dcs::test::DataPointHint<int32_t>{"TOF_HVSTATUS_SM[00..01]MOD[0..1]", 0, 524287});
  return specs;
}
```

A concrete example can be found in the `Detectors/TOF/calibration/testWorkflow` dir : `tof-calibration-dcs-sim-workflow`.

# dcs-proxy

It is the proxy to connect to the DCS machine.
For test purposes, you can run with either hard-coded DPs (--test-mode), or reading a configuration entry from CCDB, which can be created with `testWorkflow/macros/makeCCDBEntryForDCS.C`. To validate the retrieval of data, you can attach the workflow `o2-dcs-data-client`, e.g.:

```
o2-dcs-proxy --dcs-proxy '--channel-config "name=dcs-proxy,type=pull,method=connect,address=tcp://10.11.28.22:60000,rateLogging=1,transport=zeromq"' --ccdb-url http://localhost:8080 --detector-list COMMON,COMMON1 -b | o2-dcs-data-client -b
```





# dcs-config proxy

This is a proxy to import the detector configuration files from DCS server into the DPL. A simple test is

```
DET="TOF"
CHANFROM='"type=sub,method=connect,address=tcp://127.0.0.1:5556,rateLogging=1,transport=zeromq"'
CHANACK='"type=push,method=connect,address=tcp://127.0.0.1:5557,rateLogging=1,transport=zeromq"'
o2-dcs-config-proxy --subscribe-to $CHANFROM --acknowlege-to $CHANACK | o2-dcs-config-consumer-test-workflow --detector $DET
```

to receive from the `CHANFROM` DCS channel the configuration file name (starting with detector name) and the file itself and inject them as DPL messages with specs
`<DET>/DCS_CONFIG_NAME/0` and `<DET>/DCS_CONFIG_FILE/0` respectively.
The `o2-dcs-config-consumer-test-workflow` is a dummy processing device which just consumes such messages for the detector `<DET>`.

If the `CHANACK` string is not empty, then the acknowledgment string `OK` will be sent to this channel on every reception of the DCS message.

While the real exchange will be with the DCS server, for the local tests one can use the `DCS server emulator`. As a prerequisite, this will require building the `ccpzmq` package which is in the alidist but is not dependency of the `O2`:
```
cd ~/alice/
aliBuild build cppzmq --defaults o2
```

Then, one should load it together with O2, e.g. `alienv load cppzmq/latest O2/latest` and compile locally the emulator code, e.g.:
```
mkdir ~/tstDCS
cd ~/tstDCS
cp ~/alice/O2/Detectors/DCS/testWorkflow/src/dcs*.cpp ./
cp ~/alice/O2/Detectors/DCS/testWorkflow/src/compile-dcs-emulator.sh ./
source ./compile-dcs-emulator.sh
```
This will compile two executables `dcssend` and `dcsclient`. The former one is the `DCS server emulator` which has the following options:
```
./dcssend -h
```
You can use it as e.g.:
```
echo "blabla" > TOFfile.txt  # this is the file you want to send from the DCS to the config processor
xterm -e "alienv load cppzmq/latest O2/latest; ./dcssend -f TOFfile.txt -o 5556 -a 5557"& # run the server emulator in separate terminal
```

Then, in other terminal you can run your DCS config processor, as described above (make sure the ports of sender and receiver are consistent.
In case of problems you can validate the receiving process using `dcsclient` test utility (emulates `o2-dcs-config-proxy ...` workflow by receiving the file from the `DCS server` and sending it an acknowledgment):
```
./dcsclient -o 5556 -a 5557
```
