// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRD_TRACK_READER_H
#define O2_TRD_TRACK_READER_H

/// @file   TRDTrackReaderSpec.h

#include "DataFormatsTRD/TrackTRD.h"
#include "DataFormatsTRD/TrackTriggerRecord.h"

#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

#include "TFile.h"
#include "TTree.h"

#include <vector>
#include <string>
#include <memory>

using namespace o2::framework;

namespace o2
{
namespace trd
{

class TRDTrackReader : public Task
{
 public:
  enum Mode : int {
    ITSTPCTRD,
    TPCTRD
  };

  TRDTrackReader(bool useMC, Mode mode) : mUseMC(useMC), mMode(mode) {}
  ~TRDTrackReader() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  void connectTree(const std::string& filename);
  bool mUseMC = false;
  Mode mMode;
  std::unique_ptr<TFile> mFile;
  std::unique_ptr<TTree> mTree;
  std::string mFileName = "";
  std::vector<o2::trd::TrackTRD> mTracks, *mTracksPtr = &mTracks;
  std::vector<o2::trd::TrackTriggerRecord> mTrigRec, *mTrigRecPtr = &mTrigRec;
};

/// read TPC-TRD matched tracks from a root file
framework::DataProcessorSpec getTRDTPCTrackReaderSpec(bool useMC);

/// read ITS-TPC-TRD matched tracks from a root file
framework::DataProcessorSpec getTRDGlobalTrackReaderSpec(bool useMC);

} // namespace trd
} // namespace o2

#endif
