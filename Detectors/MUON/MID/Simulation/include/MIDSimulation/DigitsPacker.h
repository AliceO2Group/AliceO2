// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/DigitsPacker.h
/// \brief  Digits sorter for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#ifndef O2_MID_DIGITSPACKER_H
#define O2_MID_DIGITSPACKER_H

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"

namespace o2
{
namespace mid
{
class DigitsPacker
{
 public:
  void process(const std::vector<ColumnDataMC>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, unsigned int timestampdiff = 0);

  /// Gets the number of groups
  size_t getNGroups() const { return mTimeGroups.size() - 1; }

  bool getGroup(size_t igroup, std::vector<ColumnDataMC>& digitStore, o2::dataformats::MCTruthContainer<MCLabel>& mcContainer);

 private:
  std::vector<size_t> mTimeGroups;                                          // Time groups
  const std::vector<ColumnDataMC>* mDigitStore = nullptr;                   // Digits store (not owner)
  const o2::dataformats::MCTruthContainer<MCLabel>* mMCContainer = nullptr; // Vector of labels (not owner)
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_DIGITSPACKER_H */
