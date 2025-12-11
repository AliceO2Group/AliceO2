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

#include <vector>

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "TRKWorkflow/TrackerSpec.h"

namespace o2
{
using namespace framework;
namespace trk
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

TrackerDPL::TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       o2::gpu::gpudatatypes::DeviceType dType)
{
  // mITSTrackingInterface.setTrackingMode(trMode);
}

void TrackerDPL::init(InitContext& ic)
{
  // mTimer.Stop();
  // mTimer.Reset();
  // o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  // mChainITS.reset(mRecChain->AddChain<o2::gpu::GPUChainITS>());
  // mITSTrackingInterface.setTraitsFromProvider(mChainITS->GetITSVertexerTraits(),
  //                                             mChainITS->GetITSTrackerTraits(),
  //                                             mChainITS->GetITSTimeframe());
}

void TrackerDPL::stop()
{
  LOGF(info, "CPU Reconstruction total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

void TrackerDPL::run(ProcessingContext& pc)
{
  auto cput = mTimer.CpuTime();
  auto realt = mTimer.RealTime();
  mTimer.Start(false);
  // mITSTrackingInterface.updateTimeDependentParams(pc);
  // mITSTrackingInterface.run(pc);
  mTimer.Stop();
  LOGP(info, "CPU Reconstruction time for this TF {} s (cpu), {} s (wall)", mTimer.CpuTime() - cput, mTimer.RealTime() - realt);
}

// void TrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
// {
//   // mITSTrackingInterface.finaliseCCDB(matcher, obj);
// }

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "TRK CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC, o2::gpu::gpudatatypes::DeviceType dType)
{
  std::vector<InputSpec> inputs;

  // inputs.emplace_back("compClusters", "TRK", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  // inputs.emplace_back("patterns", "TRK", "PATTERNS", 0, Lifetime::Timeframe);
  // inputs.emplace_back("ROframes", "TRK", "CLUSTERSROF", 0, Lifetime::Timeframe);

  // inputs.emplace_back("itscldict", "TRK", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  // inputs.emplace_back("itsalppar", "TRK", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                          // orbitResetTime
                                                              false,                          // GRPECS=true
                                                              false,                          // GRPLHCIF
                                                              false,                          // GRPMagField
                                                              false,                          // askMatLUT
                                                              o2::base::GRPGeomRequest::None, // geometry, but ignored until it will be put in the CCDB
                                                              inputs,
                                                              true);
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TRK", "TRACKS", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "TRACKCLSID", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "TRKTrackROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICES", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "VERTICESROF", 0, Lifetime::Timeframe);
  // outputs.emplace_back("TRK", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    // inputs.emplace_back("trkmclabels", "TRK", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    // inputs.emplace_back("TRKMC2ROframes", "TRK", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "VERTICESMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "VERTICESMCPUR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "TRACKSMCTR", 0, Lifetime::Timeframe);
    // outputs.emplace_back("TRK", "TRKTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "trk-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                            useMC,
                                            dType)},
    Options{}};
}

} // namespace trk
} // namespace o2
