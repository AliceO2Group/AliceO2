// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "StrangenessTrackingWorkflow/HypertrackingSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/TrackReaderSpec.h"
#include "ITSMFTWorkflow/ClusterReaderSpec.h"
#include "GlobalTrackingWorkflowReaders/SecondaryVertexReaderSpec.h"

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"

#include "StrangenessTracking/HyperTracker.h"

namespace o2
{
using namespace o2::framework;
namespace strangeness_tracking
{

class HypertrackerSpec : public framework::Task
{
 public:
  HypertrackerSpec(bool isMC = false);
  ~HypertrackerSpec() override = default;

  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  bool mIsMC = false;
  TStopwatch mTimer;
  HyperTracker mTracker;
};

framework::WorkflowSpec getWorkflow(bool useMC, bool useRootInput)
{
  framework::WorkflowSpec specs;
  if (useRootInput) {
    specs.emplace_back(o2::itsmft::getITSClusterReaderSpec(useMC, true));
    specs.emplace_back(o2::its::getITSTrackReaderSpec(useMC));
    specs.emplace_back(o2::vertexing::getSecondaryVertexReaderSpec());
  }
  specs.emplace_back(getHyperTrackerSpec());
  return specs;
}

HypertrackerSpec::HypertrackerSpec(bool isMC) : mIsMC{isMC}
{
  // no ops
}

void HypertrackerSpec::init(framework::InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();

  // load geometry
  base::GeometryManager::loadGeometry();
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::L2G));

  LOG(info) << "Initialized Hypertracker...";
}

void HypertrackerSpec::run(framework::ProcessingContext& pc)
{
  mTimer.Start(false);
  LOG(info) << "Running Hypertracker...";
  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  auto patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");
  auto itsTracks = pc.inputs().get<gsl::span<o2::its::TrackITS>>("ITSTrack");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  mTimer.Stop();
}

void HypertrackerSpec::endOfStream(framework::EndOfStreamContext& ec)
{
  LOGF(info, "Hypertracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getHyperTrackerSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ITSTrack", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("vos", "GLO", "V0S", 0, Lifetime::Timeframe);                // found V0s
  inputs.emplace_back("v02pvrf", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);    // prim.vertex -> V0s refs
  inputs.emplace_back("cascs", "GLO", "CASCS", 0, Lifetime::Timeframe);            // found Cascades
  inputs.emplace_back("cas2pvrf", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "hypertracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<HypertrackerSpec>()},
  };
}

} // namespace strangeness_tracking
} // namespace o2