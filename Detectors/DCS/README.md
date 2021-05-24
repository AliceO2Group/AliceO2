<!-- doxy
\page refDetectorsDCS DCS
/doxy -->

# Export of DCS to CCDB 

To be written

# Generating DCS aliases

For test purposes, DCS aliases can be generated making use of the helper 
function `generateRandomDataPoints`. For example : 

```c++
#include "DetectorsDCS/DataPointGenerator.h"
std::vector<std::string> patterns = { "DET/HV/Crate[0.9]/Channel[00.42]/vMon" };
auto dps = o2::dcs::generateRandomDataPoints(patterns,0.0,1200.0);
```

would generate 420 data points.

# Example of DCS processing

See README in https://github.com/AliceO2Group/AliceO2/tree/dev/Detectors/TOF/calibration/testWorkflow

<!-- doxy
* \subpage refDetectorsDCStestWorkflow
/doxy -->
