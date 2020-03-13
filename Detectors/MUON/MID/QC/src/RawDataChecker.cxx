// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/QC/src/RawDataChecker.cxx
/// \brief  Class to check the raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   9 December 2019

#include "MIDQC/RawDataChecker.h"

#include <unordered_map>
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

void RawDataChecker::init(const CrateMasks& crateMasks)
{
  /// Initializes the checkers
  for (uint16_t igbt = 0; igbt < crateparams::sNGBTs; ++igbt) {
    mCheckers[igbt].init(igbt, crateMasks.getMask(igbt));
  }
}

bool RawDataChecker::process(gsl::span<const LocalBoardRO> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords)
{
  /// Checks the raw data

  bool isOk = true;
  mDebugMsg.clear();
  std::unordered_map<uint16_t, std::vector<ROFRecord>> rofs;
  for (auto& rof : rofRecords) {
    auto& loc = localBoards[rof.firstEntry];
    auto crateId = crateparams::getCrateId(loc.boardId);
    auto linkId = crateparams::getGBTIdFromBoardInCrate(crateparams::getLocId(loc.boardId));
    auto feeId = crateparams::makeROId(crateId, linkId);
    rofs[feeId].emplace_back(rof);
  }

  for (auto& item : rofs) {
    isOk &= mCheckers[item.first].process(localBoards, item.second, pageRecords);
    mDebugMsg += mCheckers[item.first].getDebugMessage();
  }

  return isOk;
}

unsigned int RawDataChecker::getNEventsProcessed() const
{
  /// Gets the number of processed events
  unsigned int sum = 0;
  for (auto& checker : mCheckers) {
    sum += checker.getNEventsProcessed();
  }
  return sum;
}

unsigned int RawDataChecker::getNEventsFaulty() const
{
  /// Gets the number of faulty events
  unsigned int sum = 0;
  for (auto& checker : mCheckers) {
    sum += checker.getNEventsFaulty();
  }
  return sum;
}

unsigned int RawDataChecker::getNBusyRaised() const
{
  /// Gets the number of busy raised
  unsigned int sum = 0;
  for (auto& checker : mCheckers) {
    sum += checker.getNBusyRaised();
  }
  return sum;
}

void RawDataChecker::clear()
{
  /// Clears the statistics
  for (auto& checker : mCheckers) {
    checker.clear();
  }
}

} // namespace mid
} // namespace o2
