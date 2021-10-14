/// @file   CalibrationWorkflow.cxx

#include "ITSWorkflow/CalibrationWorkflow.h"
#include "ITSWorkflow/CalibratorSpec.h"

#include "ITSWorkflow/ClustererSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITSMFTWorkflow/EntropyEncoderSpec.h"

namespace o2
{
namespace its
{
namespace calibration_workflow
{
framework::WorkflowSpec getWorkflow()
{
    framework::WorkflowSpec specs;

    specs.emplace_back(o2::its::getITSCalibratorSpec());

    return specs;
}
}
} // namespace itsmft
} // namespace o2