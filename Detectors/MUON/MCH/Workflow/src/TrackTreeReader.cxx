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

#include "TrackTreeReader.h"
#include <limits>
#include <fmt/format.h>

namespace o2::mch
{
void AssertBranch(ROOT::Internal::TTreeReaderValueBase& value)
{
  if (value.GetSetupStatus() < 0) {
    throw std::invalid_argument(fmt::format("Error {} setting up tree reader for branch {}",
                                            value.GetSetupStatus(), value.GetBranchName()));
  }
}

TrackTreeReader::TrackTreeReader(TTree* tree) : mCurrentRof{std::numeric_limits<size_t>::max()}
{
  if (!tree) {
    throw std::invalid_argument("cannot work with a null tree pointer");
  }
  mTreeReader.SetTree(tree);
  mTreeReader.Restart();
  mTreeReader.Next();
  mCurrentRof = 0;
  AssertBranch(mTracks);
  AssertBranch(mRofs);
  AssertBranch(mClusters);
}

bool TrackTreeReader::next(o2::mch::ROFRecord& rof, std::vector<o2::mch::TrackMCH>& tracks, std::vector<o2::mch::ClusterStruct>& clusters)
{
  if (mCurrentRof >= mRofs->size()) {
    if (!mTreeReader.Next()) {
      return false;
    }
    mCurrentRof = 0;
  }

  if (mRofs->empty()) {
    return false;
  }
  rof = (*mRofs)[mCurrentRof];
  tracks.clear();
  clusters.clear();
  auto& tfTracks = *mTracks;
  auto& tfClusters = *mClusters;
  tracks.insert(tracks.begin(), tfTracks.begin() + rof.getFirstIdx(), tfTracks.begin() + rof.getLastIdx() + 1);
  clusters.insert(clusters.begin(), tfClusters.begin() + rof.getFirstIdx(), tfClusters.begin() + rof.getLastIdx() + 1);
  ++mCurrentRof;
  return true;
}
} // namespace o2::mch
