// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/DigitsPacker.cxx
/// \brief  Implementation of the digits Sorter for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#include "MIDSimulation/DigitsPacker.h"

namespace o2
{
namespace mid
{
void DigitsPacker::process(const std::vector<ColumnDataMC>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, unsigned int timestampdiff)
{
  /// Groups the digits which have a timestamp difference smaller than timestamp diff
  /// \param inDigitStore Vector of input MC digits
  /// \param inMCContainer Container with MC labels for input MC digits
  /// \param timestampdiff Maximum timestamp difference between digits to be merged
  mTimeGroups.clear();
  mDigitStore = &inDigitStore;
  mMCContainer = &inMCContainer;

  int ts = -99999999;
  for (size_t idx = 0; idx < inDigitStore.size(); ++idx) {
    if (std::abs(inDigitStore[idx].getTimeStamp() - ts) > timestampdiff) {
      ts = inDigitStore[idx].getTimeStamp();
      mTimeGroups.emplace_back(idx);
    }
  }

  mTimeGroups.emplace_back(inDigitStore.size());
}

bool DigitsPacker::getGroup(size_t igroup, std::vector<ColumnDataMC>& digitStore, o2::dataformats::MCTruthContainer<MCLabel>& mcContainer)
{

  digitStore.clear();
  mcContainer.clear();

  if (igroup >= getNGroups()) {
    return false;
  }

  size_t start = mTimeGroups[igroup];
  size_t end = mTimeGroups[igroup + 1];

  digitStore.reserve(end - start);
  std::copy(mDigitStore->begin() + start, mDigitStore->begin() + end, std::back_inserter(digitStore));
  for (size_t idx = 0; idx < digitStore.size(); ++idx) {
    mcContainer.addElements(idx, mMCContainer->getLabels(start + idx));
  }

  return true;
}

} // namespace mid
} // namespace o2
