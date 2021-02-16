// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file  TrackCosmicsReaderSpec.h

#ifndef O2_GLOBAL_TRACKCOSMICS_READER
#define O2_GLOBAL_TRACKCOSMICS_READER

#include "TFile.h"
#include "TTree.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/TrackCosmics.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TrackCosmicsReader : public Task
{
 public:
  TrackCosmicsReader(bool useMC) : mUseMC(useMC) {}
  ~TrackCosmicsReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::dataformats::TrackCosmics> mTracks, *mTracksPtr = &mTracks;
  std::vector<o2::MCCompLabel> mLabels, *mLabelsPtr = &mLabels;
};

/// create a processor spec to read cosmic tracks from a root file
framework::DataProcessorSpec getTrackCosmicsReaderSpec(bool useMC);

} // namespace globaltracking
} // namespace o2

#endif /* O2_GLOBAL_TRACKCOSMICREADER */
