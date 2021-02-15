// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TOFMatchedReaderSpec.h

#ifndef O2_TOF_MATCHINFOREADER
#define O2_TOF_MATCHINFOREADER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/MatchInfoTOF.h"
#include "ReconstructionDataFormats/TrackTPCTOF.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace tof
{

class TOFMatchedReader : public o2::framework::Task
{
 public:
  TOFMatchedReader(bool useMC, bool tpcmatch, bool readTracks) : mUseMC(useMC), mTPCMatch(tpcmatch), mReadTracks(readTracks) {}
  ~TOFMatchedReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);

  bool mUseMC = false;
  bool mTPCMatch = false;
  bool mReadTracks = false;

  std::string mInFileName{"o2match_tof.root"};
  std::string mInTreeName{"matchTOF"};
  std::unique_ptr<TFile> mFile = nullptr;
  std::unique_ptr<TTree> mTree = nullptr;
  std::vector<o2::dataformats::MatchInfoTOF> mMatches, *mMatchesPtr = &mMatches;
  std::vector<o2::dataformats::TrackTPCTOF> mTracks, *mTracksPtr = &mTracks;
  std::vector<o2::MCCompLabel> mLabelTOF, *mLabelTOFPtr = &mLabelTOF;
};

/// create a processor spec
/// read matched TOF clusters from a ROOT file
framework::DataProcessorSpec getTOFMatchedReaderSpec(bool useMC, bool tpcmatch = false, bool readTracks = false);

} // namespace tof
} // namespace o2

#endif /* O2_TOF_MATCHINFOREADER */
