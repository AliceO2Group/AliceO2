// @file   ThresholdCalibrationWorkflow.cxx

#include "ITSWorkflow/ThresholdCalibrationWorkflow.h"
#include "ITSWorkflow/ThresholdCalibratorSpec.h"

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
namespace threshold_calibration_workflow
{
framework::WorkflowSpec getWorkflow()
{
  framework::WorkflowSpec specs;

  specs.emplace_back(o2::its::getITSThresholdCalibratorSpec());

  return specs;
}
} // namespace threshold_calibration_workflow
} // namespace its
} // namespace o2
