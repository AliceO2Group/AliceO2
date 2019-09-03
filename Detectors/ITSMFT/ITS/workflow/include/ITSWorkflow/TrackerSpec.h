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

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "ITStracking/Tracker.h"
#include "ITStracking/TrackerTraitsCPU.h"
#include "ITStracking/Vertexer.h"
#include "ITStracking/VertexerTraits.h"

namespace o2
{
namespace its
{

class TrackerDPL : public framework::Task
{
 public:
  TrackerDPL(bool isMC) : mIsMC{isMC} {}
  ~TrackerDPL() override = default;
  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mIsMC = false;
  TrackerTraitsCPU mTrackerTraits; //FIXME: the traits should be taken from the GPUChain
  VertexerTraits mVertexerTraits;
  std::unique_ptr<parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<Tracker> mTracker = nullptr;
  std::unique_ptr<Vertexer> mVertexer = nullptr;
};

/// create a processor spec
/// run ITS CA tracker
framework::DataProcessorSpec getTrackerSpec(bool useMC);

} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKERDPL */
