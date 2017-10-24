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

#include "ITSReconstruction/CA/IndexTableUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

const std::vector<std::pair<int, int>> IndexTableUtils::selectClusters(
    const std::array<int, Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins + 1> &indexTable,
    const std::array<int, 4> &selectedBinsRect)
{
  std::vector<std::pair<int, int>> filteredBins { };

  int phiBinsNum { selectedBinsRect[3] - selectedBinsRect[1] + 1 };

  if (phiBinsNum < 0) {

    phiBinsNum += Constants::IndexTable::PhiBins;
  }

  filteredBins.reserve(phiBinsNum);

  for (int iPhiBin { selectedBinsRect[1] }, iPhiCount { 0 }; iPhiCount < phiBinsNum;
      iPhiBin = ++iPhiBin == Constants::IndexTable::PhiBins ? 0 : iPhiBin, iPhiCount++) {

    const int firstBinIndex { IndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };

    filteredBins.emplace_back(indexTable[firstBinIndex],
        countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }

  return filteredBins;
}

}
}
}
