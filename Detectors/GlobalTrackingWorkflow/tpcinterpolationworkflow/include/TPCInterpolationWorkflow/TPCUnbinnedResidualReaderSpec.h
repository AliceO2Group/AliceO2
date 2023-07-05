// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   TPCUnbinnedResidualReaderSpec.h

#ifndef O2_TPC_UNBINNEDRESIDUAL_READER_H
#define O2_TPC_UNBINNEDRESIDUAL_READER_H

#include "TFile.h"
#include "TTree.h"
#include <vector>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "SpacePoints/TrackResiduals.h"
#include "SpacePoints/TrackInterpolation.h"

namespace o2
{
namespace tpc
{

class TPCUnbinnedResidualReader : public o2::framework::Task
{
 public:
  TPCUnbinnedResidualReader(bool trkInput) : mTrackInput(trkInput){};
  ~TPCUnbinnedResidualReader() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  void connectTree();
  bool mTrackInput = false;
  std::unique_ptr<TFile> mFileIn;
  std::unique_ptr<TTree> mTreeIn;
  std::string mInFileName;
  std::string mInTreeName;
  std::vector<UnbinnedResid> mUnbinnedResid, *mUnbinnedResidPtr = &mUnbinnedResid;
  std::vector<TrackData> mTrackData, *mTrackDataPtr = &mTrackData;
  std::vector<TrackDataCompact> mTrackDataCompact, *mTrackDataCompactPtr = &mTrackDataCompact;
};

/// read unbinned TPC residuals and reference tracks from a root file
framework::DataProcessorSpec getUnbinnedTPCResidualsReaderSpec(bool trkInput);

} // namespace tpc
} // namespace o2

#endif /* O2_TPC_UNBINNEDRESIDUAL_READER_H */
