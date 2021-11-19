/// @file   CalibrationWorkflow.cxx

#include "ITSWorkflow/TreeMakerWorkflow.h"
#include "ITSWorkflow/TreeMaker.h"

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
namespace tree_maker_workflow
{
framework::WorkflowSpec getWorkflow()
{
    framework::WorkflowSpec specs;

    specs.emplace_back(o2::its::getITSTreeMaker());

    return specs;
}
}
} // namespace itsmft
} // namespace o2
