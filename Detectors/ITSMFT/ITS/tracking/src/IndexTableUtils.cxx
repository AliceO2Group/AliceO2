// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file IndexTableUtils.cxx
/// \brief
///

#include "ITStracking/IndexTableUtils.h"

namespace o2
{
namespace its
{

const std::vector<std::pair<int, int>> index_table_utils::selectClusters(
  const std::array<int, constants::index_table::ZBins * constants::index_table::PhiBins + 1>& indexTable,
  const std::array<int, 4>& selectedBinsRect)
{
  std::vector<std::pair<int, int>> filteredBins{};

  int phiBinsNum{selectedBinsRect[3] - selectedBinsRect[1] + 1};

  if (phiBinsNum < 0) {
    phiBinsNum += constants::index_table::PhiBins;
  }

  filteredBins.reserve(phiBinsNum);

  for (int iPhiBin{selectedBinsRect[1]}, iPhiCount{0}; iPhiCount < phiBinsNum;
       iPhiBin = ++iPhiBin == constants::index_table::PhiBins ? 0 : iPhiBin, iPhiCount++) {

    const int firstBinIndex{index_table_utils::getBinIndex(selectedBinsRect[0], iPhiBin)};

    filteredBins.emplace_back(indexTable[firstBinIndex],
                              countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }

  return filteredBins;
}
} // namespace its
} // namespace o2
