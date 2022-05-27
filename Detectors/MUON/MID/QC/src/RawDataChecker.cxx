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

/// \file   MID/QC/src/RawDataChecker.cxx
/// \brief  Class to check the raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   9 December 2019

#include "MIDQC/RawDataChecker.h"

#include <unordered_map>
#include "fmt/format.h"
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

void RawDataChecker::init(const CrateMasks& crateMasks)
{
  /// Initializes the checkers
  for (uint16_t igbt = 0; igbt < crateparams::sNGBTs; ++igbt) {
    mCheckers[igbt].setElectronicsDelay(mElectronicsDelay);
    mCheckers[igbt].init(igbt, crateMasks.getMask(igbt));
  }
}

bool RawDataChecker::process(gsl::span<const ROBoard> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords)
{
  /// Checks the raw data

  bool isOk = true;
  mDebugMsg.clear();
  std::unordered_map<uint16_t, std::vector<ROFRecord>> rofs;
  for (auto& rof : rofRecords) {
    auto& loc = localBoards[rof.firstEntry];
    auto crateId = raw::getCrateId(loc.boardId);
    auto linkId = crateparams::getGBTIdFromBoardInCrate(raw::getLocId(loc.boardId));
    auto feeId = crateparams::makeGBTUniqueId(crateId, linkId);
    rofs[feeId].emplace_back(rof);
  }

  for (auto& item : rofs) {
    isOk &= mCheckers[item.first].process(localBoards, item.second, pageRecords);
    mDebugMsg += mCheckers[item.first].getDebugMessage();
  }

  return isOk;
}

bool RawDataChecker::checkMissingLinks(bool clear)
{
  /// Checks for missing links
  if (clear) {
    mDebugMsg.clear();
  }
  bool isOk = true;
  for (auto& checker : mCheckers) {
    if (checker.getNEventsProcessed() == 0) {
      isOk = false;
      mDebugMsg += fmt::format("Missing info from GBT 0x{:02x}\n", checker.getGBTUniqueId());
    }
  }
  return isOk;
}

void RawDataChecker::setSyncTrigger(uint32_t syncTrigger)
{
  /// Sets the trigger use to verify if all data of an event where received
  for (auto& checker : mCheckers) {
    checker.setSyncTrigger(syncTrigger);
  }
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

void RawDataChecker::clear(bool all)
{
  /// Clears the statistics
  for (auto& checker : mCheckers) {
    checker.clear(all);
  }
}

} // namespace mid
} // namespace o2
