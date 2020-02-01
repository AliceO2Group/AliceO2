// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Raw/src/ELinkDecoder.cxx
/// \brief  MID CRU core decoder
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019

#include "MIDRaw/ELinkDecoder.h"

#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

bool ELinkDecoder::add(uint8_t byte, uint8_t expectedStart)
{
  /// Add next byte
  if (mBytes.empty() && (byte & 0xc0) != expectedStart) {
    return false;
  }
  mBytes.emplace_back(byte);
  if (mBytes.size() == sMinimumSize) {
    if (crateparams::isLoc(mBytes[0])) {
      // This is a local card
      uint8_t mask = getInputs();
      for (int ich = 0; ich < 4; ++ich) {
        if ((mask >> ich) & 0x1) {
          // We expect 2 bytes for the BP and 2 for the NBP
          mTotalSize += 4;
        }
      }
    }
  }
  return true;
}

void ELinkDecoder::reset()
{
  /// Reset inner objects
  mBytes.clear();
  mTotalSize = sMinimumSize;
}

uint16_t ELinkDecoder::getPattern(int cathode, int chamber) const
{
  /// Gets the pattern
  uint8_t mask = getInputs();
  if (((mask >> chamber) & 0x1) == 0) {
    return 0;
  }

  int idx = mBytes.size();
  for (int ich = 0; ich <= chamber; ++ich) {
    if ((mask >> ich) & 0x1) {
      idx -= 4;
    }
  }

  idx += 2 * cathode;

  return joinBytes(idx);
}

} // namespace mid
} // namespace o2
