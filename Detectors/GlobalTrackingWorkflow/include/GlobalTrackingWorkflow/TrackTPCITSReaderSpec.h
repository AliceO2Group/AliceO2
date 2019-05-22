// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DigitReaderSpec.h

#ifndef O2_GLOBAL_TRACKITSTPCREADER
#define O2_GLOBAL_TRACKITSTPCREADER

#include "TFile.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ReconstructionDataFormats/TrackTPCITS.h"
#include "SimulationDataFormat/MCCompLabel.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class TrackTPCITSReader : public Task
{
 public:
  TrackTPCITSReader(bool useMC) : mUseMC(useMC) {}
  ~TrackTPCITSReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  std::unique_ptr<TFile> mFile = nullptr;
  std::vector<o2::dataformats::TrackTPCITS> mTracks, *mPtracks = &mTracks;
  std::vector<o2::MCCompLabel> mTPCLabels, *mPTPCLabels = &mTPCLabels;
  std::vector<o2::MCCompLabel> mITSLabels, *mPITSLabels = &mITSLabels;
};

/// create a processor spec
/// read simulated TOF digits from a root file
framework::DataProcessorSpec getTrackTPCITSReaderSpec(bool useMC);

} // namespace globaltracking
} // namespace o2

#endif /* O2_GLOBAL_TRACKITSTPCREADER */
