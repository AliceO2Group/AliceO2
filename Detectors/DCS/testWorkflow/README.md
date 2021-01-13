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





