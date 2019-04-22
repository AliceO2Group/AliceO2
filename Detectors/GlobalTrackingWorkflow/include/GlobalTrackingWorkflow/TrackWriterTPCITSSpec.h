// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackWriterTPCITSSpec.h

#ifndef O2_TRACKWRITER_TPCITS
#define O2_TRACKWRITER_TPCITS

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include <string>

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TrackWriterTPCITS : public Task
{
 public:
  TrackWriterTPCITS(bool useMC = true) : mUseMC(useMC) {}
  ~TrackWriterTPCITS() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  bool mFinished = false;
  bool mUseMC = true;
  std::string mOutFileName = "o2match_itstpc.root";
  std::string mTreeName = "matchTPCITS";
  std::string mOutTPCITSTracksBranchName = "TPCITS";        ///< name of branch containing output matched tracks
  std::string mOutTPCMCTruthBranchName = "MatchTPCMCTruth"; ///< name of branch for output matched tracks TPC MC
  std::string mOutITSMCTruthBranchName = "MatchITSMCTruth"; ///< name of branch for output matched tracks ITS MC
};

/// create a processor spec
/// write ITS tracks a root file
framework::DataProcessorSpec getTrackWriterTPCITSSpec(bool useMC);

} // namespace globaltracking
} // namespace o2

#endif /* O2_TRACKWRITER_TPCITS */
