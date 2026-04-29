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

/// @file   TrackerSpec.h

#ifndef O2_TRK_TRACKERDPL
#define O2_TRK_TRACKERDPL

#include "DataFormatsITSMFT/TopologyDictionary.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include <oneapi/tbb/task_arena.h>

#include "ITStracking/BoundedAllocator.h"
#include "ITStracking/TrackingInterface.h"
#include "GPUDataTypesConfig.h"

#include "DetectorsBase/GRPGeomHelper.h"

#include "TStopwatch.h"

#include <nlohmann/json.hpp>

namespace o2::trk
{
class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
             bool isMC,
             const std::string& hitRecoConfig,
             gpu::gpudatatypes::DeviceType dType = gpu::gpudatatypes::DeviceType::CPU);
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  //   void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void stop() final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  std::vector<o2::its::TrackingParameters> createTrackingParamsFromConfig();
  //   std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain = nullptr;
  //   std::unique_ptr<o2::gpu::GPUChainITS> mChainITS = nullptr;
  //   std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  //   ITSTrackingInterface mITSTrackingInterface;
  std::shared_ptr<its::BoundedMemoryResource> mMemoryPool;
  std::shared_ptr<tbb::task_arena> mTaskArena;
  nlohmann::json mHitRecoConfig;
  TStopwatch mTimer;
#ifdef O2_WITH_ACTS
  bool mUseACTS = false;
#endif
};

framework::DataProcessorSpec getTrackerSpec(bool useMC, const std::string& hitRecoConfig, gpu::gpudatatypes::DeviceType dType = gpu::gpudatatypes::DeviceType::CPU);

} // namespace o2::trk
#endif /* O2_TRK_TRACKERDPL */
