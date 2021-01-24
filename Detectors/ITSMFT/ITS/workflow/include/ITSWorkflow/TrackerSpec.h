// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"
#include "CommonUtils/StringUtils.h"
#include "TStopwatch.h"

namespace o2
{
namespace its
{

class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(bool isMC, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType = o2::gpu::GPUDataTypes::DeviceType::CPU); // : mIsMC{isMC} {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  bool mIsMC = false;
  std::string mMode = "sync";
  o2::itsmft::TopologyDictionary mDict;
  std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain = nullptr;
  std::unique_ptr<parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
  TStopwatch mTimer;
};

/// create a processor spec
/// run ITS CA tracker
framework::DataProcessorSpec getTrackerSpec(bool useMC, const std::string& trModeS, o2::gpu::GPUDataTypes::DeviceType dType);

} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKERDPL */
