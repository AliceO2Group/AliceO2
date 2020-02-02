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

#include "TFile.h"

#include "GlobalTracking/MatchTPCITS.h"
#include "DataFormatsTPC/Constants.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>
#include <vector>

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TPCITSMatchingDPL : public Task
{
 public:
  TPCITSMatchingDPL(bool useMC, const std::vector<int>& tpcClusLanes)
    : mUseMC(useMC), mTPCClusLanes(tpcClusLanes) {}
  ~TPCITSMatchingDPL() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  o2::globaltracking::MatchTPCITS mMatching; // matching engine

  std::vector<int> mTPCClusLanes;
  std::array<std::vector<char>, o2::tpc::Constants::MAXSECTOR> mBufferedTPCClusters; // at the moment not used

  bool mFinished = false;
  bool mUseMC = true;
};

/// create a processor spec
framework::DataProcessorSpec getTPCITSMatchingSpec(bool useMC, const std::vector<int>& tpcClusLanes);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACKWRITER_TPCITS */
