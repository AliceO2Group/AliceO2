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

#ifndef O2_ITS_TRACKERDPL
#define O2_ITS_TRACKERDPL

#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "ITStracking/TimeFrame.h"
#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"
#include "CommonUtils/StringUtils.h"
#include "TStopwatch.h"
#include "DetectorsBase/GRPGeomHelper.h"

namespace o2
{
namespace its
{

class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC, int trgType, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType = o2::gpu::GPUDataTypes::DeviceType::CPU); // : mIsMC{isMC} {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void setClusterDictionary(const o2::itsmft::TopologyDictionary* d) { mDict = d; }

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  bool mIsMC = false;
  bool mRunVertexer = true;
  bool mCosmicsProcessing = false;
  int mUseTriggers = 0;
  std::string mMode = "sync";
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  const o2::itsmft::TopologyDictionary* mDict = nullptr;
  std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain = nullptr;
  std::unique_ptr<o2::gpu::GPUChainITS> mChainITS = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
  TStopwatch mTimer;
};

/// create a processor spec
/// run ITS CA tracker
framework::DataProcessorSpec getTrackerSpec(bool useMC, int useTrig, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType);

} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKERDPL */
