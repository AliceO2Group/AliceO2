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

/// \file FilteredRecoTF.h
/// \brief Information filtered out from single TF

#ifndef ALICEO2_FILTERED_RECO_TF_H
#define ALICEO2_FILTERED_RECO_TF_H

#include <vector>
#include <string>
#include <Rtypes.h>
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::dataformats
{

struct FilteredRecoTF {
  struct Header {
    std::uint64_t run = 0;          // run number
    std::uint64_t creationTime = 0; // creation time from the DataProcessingHeader
    std::uint32_t firstTForbit = 0; // first orbit of time frame as unique identifier within the run

    std::string asString() const;
    void clear();

    ClassDefNV(Header, 1);
  };

  Header header{};

  // here we put the blocks corresponding to different data types

  // ITS tracks
  std::vector<o2::itsmft::ROFRecord> ITSTrackROFs{};
  std::vector<o2::its::TrackITS> ITSTracks{};
  std::vector<int> ITSClusterIndices{};
  std::vector<o2::MCCompLabel> ITSTrackMCTruth{};
  // ITS clusters
  std::vector<o2::itsmft::ROFRecord> ITSClusterROFs{};
  std::vector<o2::itsmft::CompClusterExt> ITSClusters{};
  std::vector<unsigned char> ITSClusterPatterns{};

  void clear();

  ClassDefNV(FilteredRecoTF, 1);
};

} // namespace o2::dataformats

#endif // ALICEO2_FILTERED_RECO_TF_Hxs
