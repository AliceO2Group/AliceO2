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

#ifndef O2_MFT_TRACKREADER
#define O2_MFT_TRACKREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsMFT/TrackMFT.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

namespace o2
{
namespace mft
{

class TrackReader : public o2::framework::Task
{

 public:
  TrackReader(bool useMC = true);
  ~TrackReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 protected:
  void connectTree(const std::string& filename);

  std::vector<o2::itsmft::ROFRecord> mROFRec, *mROFRecInp = &mROFRec;
  std::vector<o2::mft::TrackMFT> mTracks, *mTracksInp = &mTracks;
  std::vector<int> mClusInd, *mClusIndInp = &mClusInd;
  std::vector<o2::MCCompLabel> mMCTruth, *mMCTruthInp = &mMCTruth;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginMFT;

  bool mUseMC = true; // use MC truth

  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mInputFileName = "";
  std::string mTrackTreeName = "o2sim";
  std::string mROFBranchName = "MFTTracksROF";
  std::string mTrackBranchName = "MFTTrack";
  std::string mClusIdxBranchName = "MFTTrackClusIdx";
  std::string mTrackMCTruthBranchName = "MFTTrackMCTruth";
};

/// create a processor spec
/// read MFT track data from a root file
framework::DataProcessorSpec getMFTTrackReaderSpec(bool useMC = true);

} // namespace mft
} // namespace o2

#endif /* O2_MFT_TRACKREADER */
