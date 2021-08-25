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

/// \file   MID/Filtering/src/ChannelScalers.cxx
/// \brief  MID channel scalers
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 March 2021

#include "MIDFiltering/ChannelScalers.h"

namespace o2
{
namespace mid
{
void ChannelScalers::count(uint8_t deId, uint8_t columnId, int lineId, int cathode, uint16_t pattern)
{
  /// Increases strip counter
  for (int istrip = 0; istrip < 16; ++istrip) {
    if (pattern & (1 << istrip)) {
      ++mScalers[getChannelId(deId, columnId, lineId, istrip, cathode)];
    }
  }
}

void ChannelScalers::count(const ColumnData& patterns)
{
  //// Increases strip counter
  count(patterns.deId, patterns.columnId, 0, 1, patterns.getNonBendPattern());
  for (int iline = 0; iline < 4; ++iline) {
    count(patterns.deId, patterns.columnId, iline, 0, patterns.getBendPattern(iline));
  }
}

} // namespace mid
} // namespace o2
