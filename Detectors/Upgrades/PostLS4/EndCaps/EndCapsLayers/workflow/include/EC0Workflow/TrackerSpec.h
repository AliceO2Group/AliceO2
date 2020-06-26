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

#ifndef O2_EC0_TRACKERDPL
#define O2_EC0_TRACKERDPL

#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsEndCaps/TopologyDictionary.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "EC0tracking/Tracker.h"
#include "EC0tracking/TrackerTraitsCPU.h"
#include "EC0tracking/Vertexer.h"
#include "EC0tracking/VertexerTraits.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainEC0.h"
#include "CommonUtils/StringUtils.h"
#include "TStopwatch.h"

namespace o2
{
namespace ecl
{

class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(bool isMC, o2::gpu::GPUDataTypes::DeviceType dType = o2::gpu::GPUDataTypes::DeviceType::CPU); // : mIsMC{isMC} {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  bool mIsMC = false;
  o2::endcaps::TopologyDictionary mDict;
  std::unique_ptr<o2::gpu::GPUReconstruction> mRecChain = nullptr;
  std::unique_ptr<parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
  TStopwatch mTimer;
};

/// create a processor spec
/// run EC0 CA tracker
framework::DataProcessorSpec getTrackerSpec(bool useMC, o2::gpu::GPUDataTypes::DeviceType dType);

} // namespace ecl
} // namespace o2

#endif /* O2_EC0_TRACKERDPL */
