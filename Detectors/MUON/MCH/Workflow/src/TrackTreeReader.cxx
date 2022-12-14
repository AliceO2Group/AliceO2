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
#include <algorithm>
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
  if (tree->GetBranchStatus("trackdigits")) {
    mDigits = std::make_unique<TTreeReaderValue<std::vector<o2::mch::Digit>>>(mTreeReader, "trackdigits");
  }
  if (tree->GetBranchStatus("tracklabels")) {
    mLabels = std::make_unique<TTreeReaderValue<std::vector<o2::MCCompLabel>>>(mTreeReader, "tracklabels");
  }
  mTreeReader.SetTree(tree);
  mTreeReader.Restart();
  mTreeReader.Next();
  mCurrentRof = 0;
  AssertBranch(mTracks);
  AssertBranch(mRofs);
  AssertBranch(mClusters);
  if (hasDigits()) {
    AssertBranch(*mDigits);
  }
  if (hasLabels()) {
    AssertBranch(*mLabels);
  }
}

bool TrackTreeReader::next(o2::mch::ROFRecord& rof, std::vector<o2::mch::TrackMCH>& tracks,
                           std::vector<o2::mch::Cluster>& clusters, std::vector<o2::mch::Digit>& digits,
                           std::vector<o2::MCCompLabel>& labels)
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
  digits.clear();
  labels.clear();
  auto& tfTracks = *mTracks;
  tracks.insert(tracks.begin(), tfTracks.begin() + rof.getFirstIdx(), tfTracks.begin() + rof.getLastIdx() + 1);
  if (!tracks.empty()) {
    auto& tfClusters = *mClusters;
    clusters.insert(clusters.begin(), tfClusters.begin() + tracks.front().getFirstClusterIdx(),
                    tfClusters.begin() + tracks.back().getLastClusterIdx() + 1);
    if (hasDigits()) {
      auto& tfDigits = **mDigits;
      auto firstDigitIdx = clusters.front().firstDigit;
      auto lastDigitIdx = clusters.back().firstDigit + clusters.back().nDigits - 1;
      for (auto& cluster : clusters) {
        lastDigitIdx = std::max(lastDigitIdx, cluster.firstDigit + cluster.nDigits - 1);
        cluster.firstDigit -= firstDigitIdx;
      }
      digits.insert(digits.begin(), tfDigits.begin() + firstDigitIdx, tfDigits.begin() + lastDigitIdx + 1);
    }
  }
  if (hasLabels()) {
    auto& tfLabels = **mLabels;
    labels.insert(labels.begin(), tfLabels.begin() + rof.getFirstIdx(), tfLabels.begin() + rof.getLastIdx() + 1);
  }
  ++mCurrentRof;
  return true;
}
} // namespace o2::mch
