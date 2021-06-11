// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TrackReaderSpec.h

#ifndef O2_TPC_TRACKREADER
#define O2_TPC_TRACKREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

using namespace o2::framework;

namespace o2
{
namespace tpc
{
///< DPL device to read and send the TPC tracks (+MC) info

class TrackReader : public Task
{
 public:
  TrackReader(bool useMC = true);
  ~TrackReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void accumulate(int from, int n);
  void connectTree(const std::string& filename);

  std::vector<o2::tpc::TrackTPC>*mTracksInp = nullptr, mTracksOut;
  std::vector<o2::tpc::TPCClRefElem>*mCluRefVecInp = nullptr, mCluRefVecOut;
  std::vector<o2::MCCompLabel>*mMCTruthInp = nullptr, mMCTruthOut;

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;

  bool mUseMC = true; // use MC truth

  std::string mInputFileName = "tpctracks.root";
  std::string mTrackTreeName = "tpcrec";
  std::string mTrackBranchName = "TPCTracks";
  std::string mClusRefBranchName = "ClusRefs";
  std::string mTrackMCTruthBranchName = "TPCTracksMCTruth";
};

/// create a processor spec
/// read TPC track data from a root file
framework::DataProcessorSpec getTPCTrackReaderSpec(bool useMC = true);

} // namespace tpc
} // namespace o2

#endif /* O2_TPC_TRACKREADER */
