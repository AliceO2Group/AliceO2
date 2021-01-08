// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCITSMatchingSpec.h

#ifndef O2_MATCHING_TPCITS_SPEC
#define O2_MATCHING_TPCITS_SPEC

#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsTPC/Constants.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>
#include <vector>
#include "TStopwatch.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TPCITSMatchingDPL : public Task
{
 public:
  TPCITSMatchingDPL(bool useFT0, bool calib, bool useMC) : mUseFT0(useFT0), mCalibMode(calib), mUseMC(useMC) {}
  ~TPCITSMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;
  void endOfStream(framework::EndOfStreamContext& ec) final;

 private:
  o2::globaltracking::MatchTPCITS mMatching; // matching engine
  o2::itsmft::TopologyDictionary mITSDict;   // cluster patterns dictionary
  bool mUseFT0 = false;
  bool mCalibMode = false;
  bool mUseMC = true;
  TStopwatch mTimer;
};

/// create a processor spec
framework::DataProcessorSpec getTPCITSMatchingSpec(bool useFT0, bool calib, bool useMC, const std::vector<int>& tpcClusLanes);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACKWRITER_TPCITS */
