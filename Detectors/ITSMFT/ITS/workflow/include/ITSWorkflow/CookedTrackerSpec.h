// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CookedTrackerSpec.h

#ifndef O2_ITS_COOKEDTRACKERDPL
#define O2_ITS_COOKEDTRACKERDPL

#include "Framework/DataProcessorSpec.h"
#include "ITSReconstruction/CookedTracker.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "Framework/Task.h"
#include "TStopwatch.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

class CookedTrackerDPL : public Task
{
 public:
  CookedTrackerDPL(bool useMC) : mUseMC(useMC) {}
  ~CookedTrackerDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  o2::itsmft::TopologyDictionary mDict;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  o2::its::CookedTracker mTracker;
  TStopwatch mTimer;
};

/// create a processor spec
/// run ITS CookedMatrix tracker
framework::DataProcessorSpec getCookedTrackerSpec(bool useMC);

} // namespace its
} // namespace o2

#endif /* O2_ITS_COOKEDTRACKERDPL */
