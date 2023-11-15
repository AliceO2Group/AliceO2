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

/// \file   MID/Filtering/src/FiltererBC.cxx
/// \brief  BC filterer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 January 2023

#include "MIDFiltering/FiltererBC.h"

namespace o2
{
namespace mid
{

int FiltererBC::matchedCollision(int bc)
{
  // First we check the simple case to save time
  if (mBunchFilling.testInteractingBC(bc)) {
    return bc;
  }

  // Then we check for neighbor BCs
  // We split the check in two so that we start the check from the input bc
  // and we move farther from it.

  int matchedBC = std::numeric_limits<int>::max();

  // We start on the upper direction

  // The allowed BC are o2::constants::lhc::LHCMaxBunches.
  // Numeration starts from 0, so o2::constants::lhc::LHCMaxBunches is not allowed.
  // This ensures that we do not check out of boundaries
  int end = bc + mBCDiffHigh;
  if (end >= o2::constants::lhc::LHCMaxBunches) {
    end = o2::constants::lhc::LHCMaxBunches - 1;
  }
  // We exclude the input BC since it was already checked before
  for (int ibc = bc + 1; ibc <= end; ++ibc) {
    if (mBunchFilling.testInteractingBC(ibc)) {
      // We do not stop here because we want first to check that there is no closer matching in the other direction
      matchedBC = ibc;
      break;
    }
  }

  // We then check the lower direction
  end = bc + mBCDiffLow;
  if (end < 0) {
    end = 0;
  }
  // We exclude the input BC since it was already checked before
  for (int ibc = bc - 1; ibc >= end; --ibc) {
    if (mBunchFilling.testInteractingBC(ibc)) {
      if (bc - ibc < matchedBC - bc) {
        // This collision BC is closer to the input BC than the one found in the upper side
        return ibc;
      }
      break;
    }
  }

  return matchedBC;
}

std::vector<ROFRecord> FiltererBC::process(gsl::span<const ROFRecord> rofRecords)
{
  std::vector<ROFRecord> filteredROFs;
  auto rofIt = rofRecords.begin();
  auto end = rofRecords.end();
  // Loop on ROFs
  for (; rofIt != end; ++rofIt) {
    // Check if BC matches a collision BC
    auto matchedColl = matchedCollision(rofIt->interactionRecord.bc);
    if (matchedColl < o2::constants::lhc::LHCMaxBunches) {
      // Add this ROF to the filtered ones
      filteredROFs.emplace_back(*rofIt);
      if (mSelectOnly) {
        continue;
      }
      // Use the BC of the matching collision
      filteredROFs.back().interactionRecord.bc = matchedColl;
      // Search for neighbor BCs
      for (auto auxIt = rofIt + 1; auxIt != end; ++auxIt) {
        // We assume that data are time-ordered
        if (auxIt->interactionRecord.orbit != rofIt->interactionRecord.orbit) {
          // Orbit does not match
          break;
        }
        int bcDiff = auxIt->interactionRecord.bc - matchedColl;
        if (bcDiff < mBCDiffLow || bcDiff > mBCDiffHigh) {
          // BC is not in the allowed window
          break;
        }
        filteredROFs.back().nEntries += auxIt->nEntries;
        // CAVEAT: we are updating rofIt here. Do not use it in the following
        rofIt = auxIt;
      }
    }
  }
  return filteredROFs;
}

} // namespace mid
} // namespace o2
