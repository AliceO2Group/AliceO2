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
#include "ITSWorkflow/TrackerSpec.h"

namespace o2
{
using namespace framework;
namespace its
{
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

TrackerDPL::TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
                       bool isMC,
                       int trgType,
                       const TrackingMode& trMode,
                       const bool overrBeamEst,
                       o2::gpu::GPUDataTypes::DeviceType dType) : mGGCCDBRequest(gr),
                                                                  mRecChain{o2::gpu::GPUReconstruction::CreateInstance(dType, true)},
                                                                  mITSTrackingInterface{isMC, trgType, overrBeamEst}
{
  mITSTrackingInterface.setTrackingMode(trMode);
}

void TrackerDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mTimer.Reset();
  o2::base::GRPGeomHelper::instance().setRequest(mGGCCDBRequest);
  mChainITS.reset(mRecChain->AddChain<o2::gpu::GPUChainITS>());
  mITSTrackingInterface.setTraitsFromProvider(mChainITS->GetITSVertexerTraits(),
                                              mChainITS->GetITSTrackerTraits(),
                                              mChainITS->GetITSTimeframe());
  mITSTrackingInterface.initialise();
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
  mITSTrackingInterface.updateTimeDependentParams(pc);
  mITSTrackingInterface.run(pc);
  mTimer.Stop();
  LOGP(info, "CPU Reconstruction time for this TF {} s (cpu), {} s (wall)", mTimer.CpuTime() - cput, mTimer.RealTime() - realt);
}

void TrackerDPL::finaliseCCDB(ConcreteDataMatcher& matcher, void* obj)
{
  mITSTrackingInterface.finaliseCCDB(matcher, obj);
}

void TrackerDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "ITS CA-Tracker total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getTrackerSpec(bool useMC, bool useGeom, int trgType, const std::string& trModeS, const bool overrBeamEst, o2::gpu::GPUDataTypes::DeviceType dType)
{
  std::vector<InputSpec> inputs;

  inputs.emplace_back("compClusters", "ITS", "COMPCLUSTERS", 0, Lifetime::Timeframe);
  inputs.emplace_back("patterns", "ITS", "PATTERNS", 0, Lifetime::Timeframe);
  inputs.emplace_back("ROframes", "ITS", "CLUSTERSROF", 0, Lifetime::Timeframe);
  if (trgType == 1) {
    inputs.emplace_back("phystrig", "ITS", "PHYSTRIG", 0, Lifetime::Timeframe);
  } else if (trgType == 2) {
    inputs.emplace_back("phystrig", "TRD", "TRKTRGRD", 0, Lifetime::Timeframe);
  }
  inputs.emplace_back("itscldict", "ITS", "CLUSDICT", 0, Lifetime::Condition, ccdbParamSpec("ITS/Calib/ClusterDictionary"));
  inputs.emplace_back("itsalppar", "ITS", "ALPIDEPARAM", 0, Lifetime::Condition, ccdbParamSpec("ITS/Config/AlpideParam"));
  auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false,                                                                        // orbitResetTime
                                                              true,                                                                         // GRPECS=true
                                                              false,                                                                        // GRPLHCIF
                                                              true,                                                                         // GRPMagField
                                                              true,                                                                         // askMatLUT
                                                              useGeom ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None, // geometry
                                                              inputs,
                                                              true);
  if (!useGeom) {
    ggRequest->addInput({"itsTGeo", "ITS", "GEOMTGEO", 0, Lifetime::Condition, framework::ccdbParamSpec("ITS/Config/Geometry")}, inputs);
  }
  if (overrBeamEst) {
    inputs.emplace_back("meanvtx", "GLO", "MEANVERTEX", 0, Lifetime::Condition, ccdbParamSpec("GLO/Calib/MeanVertex", {}, 1));
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("ITS", "TRACKS", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "TRACKCLSID", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "ITSTrackROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICES", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "VERTICESROF", 0, Lifetime::Timeframe);
  outputs.emplace_back("ITS", "IRFRAMES", 0, Lifetime::Timeframe);

  if (useMC) {
    inputs.emplace_back("itsmclabels", "ITS", "CLUSTERSMCTR", 0, Lifetime::Timeframe);
    inputs.emplace_back("ITSMC2ROframes", "ITS", "CLUSTERSMC2ROF", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "VERTICESMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "TRACKSMCTR", 0, Lifetime::Timeframe);
    outputs.emplace_back("ITS", "ITSTrackMC2ROF", 0, Lifetime::Timeframe);
  }

  return DataProcessorSpec{
    "its-tracker",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TrackerDPL>(ggRequest,
                                            useMC,
                                            trgType,
                                            trModeS == "sync" ? o2::its::TrackingMode::Sync : trModeS == "async" ? o2::its::TrackingMode::Async
                                                                                                                 : o2::its::TrackingMode::Cosmics,
                                            overrBeamEst,
                                            dType)},
    Options{}};
}

} // namespace its
} // namespace o2
