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

/// \file FilteredRecoTF.cxx
/// \brief Information filtered out from single TF

#include "DataFormatsGlobalTracking/FilteredRecoTF.h"
#include <fmt/printf.h>
#include <iostream>
#include "CommonUtils/StringUtils.h"

using namespace o2::dataformats;

std::string FilteredRecoTF::Header::asString() const
{
  return fmt::format("Run:{}, TF 1st orbit:{}, creation time:{}", run, firstTForbit, creationTime);
}

void FilteredRecoTF::Header::clear()
{
  run = 0;
  creationTime = 0;
  firstTForbit = 0;
}

void FilteredRecoTF::clear()
{
  header.clear();
  //
  ITSTrackROFs.clear();
  ITSTracks.clear();
  ITSClusterIndices.clear();
  ITSTrackMCTruth.clear();
  // ITS clusters
  ITSClusterROFs.clear();
  ITSClusters.clear();
  ITSClusterPatterns.clear();
}
