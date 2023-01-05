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

#ifndef O2_MCH_WORKFLOW_TRACK_TREE_READER_H
#define O2_MCH_WORKFLOW_TRACK_TREE_READER_H

#include "DataFormatsMCH/Cluster.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/TrackMCH.h"
#include "DataFormatsMCH/ROFRecord.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <TTreeReader.h>
#include <memory>
#include <vector>

namespace o2::mch
{

class TrackTreeReader
{
 public:
  TrackTreeReader(TTree* tree);

  bool next(ROFRecord& rof,
            std::vector<TrackMCH>& tracks,
            std::vector<Cluster>& clusters,
            std::vector<Digit>& digits,
            std::vector<o2::MCCompLabel>& labels);

  bool hasDigits() { return mDigits.get() != nullptr; }
  bool hasLabels() { return mLabels.get() != nullptr; }

 private:
  TTreeReader mTreeReader;
  TTreeReaderValue<std::vector<o2::mch::TrackMCH>> mTracks = {mTreeReader, "tracks"};
  TTreeReaderValue<std::vector<o2::mch::ROFRecord>> mRofs = {mTreeReader, "trackrofs"};
  TTreeReaderValue<std::vector<o2::mch::Cluster>> mClusters = {mTreeReader, "trackclusters"};
  std::unique_ptr<TTreeReaderValue<std::vector<o2::mch::Digit>>> mDigits{};
  std::unique_ptr<TTreeReaderValue<std::vector<o2::MCCompLabel>>> mLabels{};
  size_t mCurrentRof;
};
} // namespace o2::mch
#endif
