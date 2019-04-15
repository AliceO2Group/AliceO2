// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/DigitsMerger.h
/// \brief  Digits merger for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2018
#ifndef O2_MID_DIGITSMERGER_H
#define O2_MID_DIGITSMERGER_H

#include <vector>
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDSimulation/ColumnDataMC.h"
#include "MIDSimulation/MCLabel.h"

namespace o2
{
namespace mid
{
class DigitsMerger
{
 public:
  void process(const std::vector<ColumnDataMC>& inDigitStore, const o2::dataformats::MCTruthContainer<MCLabel>& inMCContainer, std::vector<ColumnData>& outDigitStore, o2::dataformats::MCTruthContainer<MCLabel>& outMCContainer);

 private:
  std::vector<std::pair<ColumnDataMC, std::vector<size_t>>> mDigitsLabels; //! Temporary digits store
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_DIGITSMERGER_H */
