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
#include "Framework/Task.h"

using namespace o2::framework;

namespace o2
{
namespace ITS
{

class CookedTrackerDPL : public Task
{
 public:
  CookedTrackerDPL() = default;
  ~CookedTrackerDPL() = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  o2::ITS::CookedTracker mTracker;
};

/// create a processor spec
/// run ITS CookedMatrix tracker
framework::DataProcessorSpec getCookedTrackerSpec();

} // namespace ITS
} // namespace o2

#endif /* O2_ITS_COOKEDTRACKERDPL */
