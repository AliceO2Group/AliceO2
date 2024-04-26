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

#ifndef O2_ITS3_TRACKERDPL
#define O2_ITS3_TRACKERDPL

#include "DataFormatsParameters/GRPObject.h"
#include "ITS3Reconstruction/TopologyDictionary.h"

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

namespace o2::its3
{

class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, bool isMC, int trgType, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType = o2::gpu::GPUDataTypes::DeviceType::CPU);
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;
  void setClusterDictionary(o2::its3::TopologyDictionary* d) { mDict = d; }

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);

  bool mIsMC{false};
  bool mRunVertexer{true};
  bool mCosmicsProcessing{false};
  int mUseTriggers{0};
  std::string mMode{"sync"};
  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest{};
  o2::its3::TopologyDictionary* mDict{};
  std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain{};
  std::unique_ptr<o2::gpu::GPUChainITS> mChainITS{};
  std::unique_ptr<its::Tracker> mTracker{};
  std::unique_ptr<its::Vertexer> mVertexer{};
  TStopwatch mTimer{};
};

/// create a processor spec
/// run ITS CA tracker
framework::DataProcessorSpec getTrackerSpec(bool useMC, bool useGeom, int useTrig, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType);

} // namespace o2::its3

#endif /* O2_ITS_TRACKERDPL */
