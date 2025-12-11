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

#include "ITStracking/TrackingInterface.h"
#include "GPUDataTypesConfig.h"

#include "DetectorsBase/GRPGeomHelper.h"

#include "TStopwatch.h"

namespace o2::trk
{
class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
             bool isMC,
             gpu::gpudatatypes::DeviceType dType = gpu::gpudatatypes::DeviceType::CPU);
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  //   void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void stop() final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  //   std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain = nullptr;
  //   std::unique_ptr<o2::gpu::GPUChainITS> mChainITS = nullptr;
  //   std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  //   ITSTrackingInterface mITSTrackingInterface;
  TStopwatch mTimer;
};

framework::DataProcessorSpec getTrackerSpec(bool useMC, gpu::gpudatatypes::DeviceType dType = gpu::gpudatatypes::DeviceType::CPU);

} // namespace o2::trk
#endif /* O2_TRK_TRACKERDPL */
