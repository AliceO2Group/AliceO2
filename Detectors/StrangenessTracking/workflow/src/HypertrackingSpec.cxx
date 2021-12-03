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

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
using namespace o2::framework;
namespace strangeness_tracking
{

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

void HypertrackerDPL::init(framework::InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();

  LOG(info) << "Initialized HypertrackerDPL";
}

void HypertrackerDPL::run(framework::ProcessingContext& pc)
{
  mTimer.Start(false);
  LOG(info) << "Running HypertrackerDPL";
  auto compClusters = pc.inputs().get<gsl::span<o2::itsmft::CompClusterExt>>("compClusters");
  gsl::span<const unsigned char> patterns = pc.inputs().get<gsl::span<unsigned char>>("patterns");

  // code further down does assignment to the rofs and the altered object is used for output
  // we therefore need a copy of the vector rather than an object created directly on the input data,
  // the output vector however is created directly inside the message memory thus avoiding copy by
  // snapshot
  auto rofsinput = pc.inputs().get<gsl::span<o2::itsmft::ROFRecord>>("ROframes");
  mTimer.Stop();
}

void HypertrackerDPL::endOfStream(framework::EndOfStreamContext& ec)
{
  LOGF(info, "Hypertracker total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getHyperTrackerSpec()
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ITSTrack", "ITS", "TRACKS", 0, Lifetime::Timeframe);
  inputs.emplace_back("asd1", "GLO", "V0S", 0, Lifetime::Timeframe);           // found V0s
  inputs.emplace_back("asd2", "GLO", "PVTX_V0REFS", 0, Lifetime::Timeframe);   // prim.vertex -> V0s refs
  inputs.emplace_back("asd3", "GLO", "CASCS", 0, Lifetime::Timeframe);         // found Cascades
  inputs.emplace_back("asd4", "GLO", "PVTX_CASCREFS", 0, Lifetime::Timeframe); // prim.vertex -> Cascades refs
  // inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  // inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "hypertracker",
    inputs,
    outputs};
}

} // namespace strangeness_tracking
} // namespace o2