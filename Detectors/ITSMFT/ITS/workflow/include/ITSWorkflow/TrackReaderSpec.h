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

#ifndef O2_ITS_TRACKREADER
#define O2_ITS_TRACKREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Headers/DataHeader.h"
#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsITSMFT/ROFRecord.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

class TrackReader : public Task
{
 public:
  TrackReader(bool useMC = true);
  ~TrackReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 protected:
  void accumulate();

  std::vector<o2::itsmft::ROFRecord>*mROFRecInp = nullptr, mROFRecOut;
  std::vector<o2::its::TrackITS>*mTracksInp = nullptr, mTracksOut;
  std::vector<int>*mClusIndInp = nullptr, mClusIndOut;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>*mMCTruthInp = nullptr, mMCTruthOut;

  o2::header::DataOrigin mOrigin = o2::header::gDataOriginITS;

  bool mFinished = false;
  bool mUseMC = true; // use MC truth

  std::string mInputFileName = "";
  std::string mTrackTreeName = "o2sim";
  std::string mROFTreeName = "ITSTracksROF";
  std::string mTrackBranchName = "ITSTrack";
  std::string mClusIdxBranchName = "ITSTrackClusIdx";
  std::string mTrackMCTruthBranchName = "ITSTrackMCTruth";
};

/// create a processor spec
/// read ITS track data from a root file
framework::DataProcessorSpec getITSTrackReaderSpec(bool useMC = true);

} // namespace its
} // namespace o2

#endif /* O2_ITS_TRACKREADER */
