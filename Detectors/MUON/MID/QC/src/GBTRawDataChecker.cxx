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

/// \file   MID/QC/src/GBTRawDataChecker.cxx
/// \brief  Class to check the raw data from a GBT link
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   28 April 2020

#include "MIDQC/GBTRawDataChecker.h"

#include <sstream>
#include <fmt/format.h>
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

void GBTRawDataChecker::init(uint16_t gbtUniqueId, uint8_t mask)
{
  /// Initializer
  mGBTUniqueId = gbtUniqueId;
  mCrateMask = mask;
}

bool GBTRawDataChecker::checkLocalBoardSize(const ROBoard& board)
{
  /// Checks that the board has the expected non-null patterns

  // This test only make sense when we have a self-trigger,
  // since in this case we expect to have a variable number of non-zero pattern
  // as indicated by the corresponding word.
  for (int ich = 0; ich < 4; ++ich) {
    bool isExpectedNull = (((board.firedChambers >> ich) & 0x1) == 0);
    bool isNull = (board.patternsBP[ich] == 0 && board.patternsNBP[ich] == 0);
    if (isExpectedNull != isNull) {
      std::stringstream ss;
      ss << "wrong size for local board:\n";
      ss << board << "\n";
      mEventDebugMsg += ss.str();
      return false;
    }
  }
  return true;
}

bool GBTRawDataChecker::checkLocalBoardSize(const std::vector<ROBoard>& boards)
{
  /// Checks that the boards have the expected non-null patterns
  for (auto& board : boards) {
    if (!checkLocalBoardSize(board)) {
      return false;
    }
  }
  return true;
}

bool GBTRawDataChecker::checkConsistency(const ROBoard& board)
{
  /// Checks that the event information is consistent

  bool isSoxOrReset = board.triggerWord & (raw::sSOX | raw::sEOX | raw::sRESET);
  bool isCalib = raw::isCalibration(board.triggerWord);
  bool isPhys = board.triggerWord & raw::sPHY;

  // FIXME: During data acquisition we do not expect a calibration trigger
  // in coincidence with a physics trigger.
  // However, this situation can happen in the tests with the LTU.
  // So, let us remove these tests for the time being

  // if (isPhys) {
  //   if (isCalib) {
  //     mEventDebugMsg += "inconsistent trigger: calibration and physics trigger cannot be fired together\n";
  //     return false;
  //   }
  //   if (raw::isLoc(board.statusWord)) {
  //     if (board.firedChambers) {
  //       mEventDebugMsg += "inconsistent trigger: fired chambers should be 0\n";
  //       return false;
  //     }
  //   }
  // }
  if (isSoxOrReset && (isCalib || isPhys)) {
    mEventDebugMsg += "inconsistent trigger: cannot be SOX and calibration\n";
    return false;
  }

  return true;
}

bool GBTRawDataChecker::checkConsistency(const std::vector<ROBoard>& boards)
{
  /// Checks that the event information is consistent
  for (auto& board : boards) {
    if (!checkConsistency(board)) {
      std::stringstream ss;
      ss << board << "\n";
      mEventDebugMsg += ss.str();
      return false;
    }
  }
  return true;
}

bool GBTRawDataChecker::checkMasks(const std::vector<ROBoard>& locs)
{
  /// Checks the masks
  for (auto loc : locs) {
    // The board patterns coincide with the masks ("overwritten" mode)
    if (loc.statusWord & raw::sOVERWRITTEN) {
      auto maskItem = mMasks.find(loc.boardId);
      for (int ich = 0; ich < 4; ++ich) {
        uint16_t maskBP = 0;
        uint16_t maskNBP = 0;
        if (maskItem != mMasks.end()) {
          maskBP = maskItem->second.patternsBP[ich];
          maskNBP = maskItem->second.patternsNBP[ich];
        }
        if (maskBP != loc.patternsBP[ich] || maskNBP != loc.patternsNBP[ich]) {
          std::stringstream ss;
          ss << "Pattern is not compatible with mask for:\n";
          ss << loc << "\n";
          mEventDebugMsg += ss.str();
          return false;
        }
      }
    }
  }
  return true;
}

bool GBTRawDataChecker::checkRegLocConsistency(const std::vector<ROBoard>& regs, const std::vector<ROBoard>& locs, const InteractionRecord& ir)
{
  /// Checks consistency between local and regional info
  uint8_t regFired{0};
  bool isTrig = false;
  for (auto& reg : regs) {
    uint8_t ireg = raw::getLocId(reg.boardId) % 2;
    if (reg.triggerWord == 0) {
      // Self-triggered event: check the decision
      regFired |= (reg.firedChambers << (4 * ireg));
    } else {
      // Triggered event: all active boards must answer
      regFired |= (mCrateMask & (0xF << (4 * ireg)));
      isTrig = true;
    }
  }
  uint8_t locFired{0};
  for (auto& loc : locs) {
    auto linkId = getElinkId(loc);
    uint8_t mask = (1 << linkId);
    if (loc.triggerWord == 0) {
      // Self-triggered event: check the decision
      if (loc.firedChambers) {
        locFired |= mask;
      }
    } else {
      // Triggered event: all active boards must answer
      locFired |= mask;
      isTrig = true;
    }
  }

  // The XOR returns 1 in case of a difference
  uint8_t problems = (regFired ^ locFired);

  if (problems) {
    // It can be that a busy signal was raised by one of the board in previous events
    // If the board is still busy it will not answer.
    uint8_t busy{0};
    for (uint8_t iboard = 0; iboard < crateparams::sNELinksPerGBT; ++iboard) {
      auto rawIr = getRawIR(iboard, isTrig, ir);
      bool isBusy = false;
      for (auto& busyInfo : mBusyPeriods[iboard]) {
        if (busyInfo.interactionRecord > rawIr) {
          break;
        }
        isBusy = busyInfo.isBusy;
      }
      if (isBusy) {
        busy |= (iboard < crateparams::sMaxNBoardsInLink) ? (1 << iboard) : (0xF << (4 * (iboard % 2)));
      }
    }

    if (problems & ~busy) {
      std::stringstream ss;
      ss << fmt::format("loc-reg inconsistency: fired locals ({:08b}) != expected from reg ({:08b});\n", locFired, regFired);
      ss << printBoards(regs);
      ss << printBoards(locs);
      mEventDebugMsg += ss.str();
      return false;
    }
  }
  return true;
}

std::string GBTRawDataChecker::printBoards(const std::vector<ROBoard>& boards) const
{
  /// Prints the boards
  std::stringstream ss;
  for (auto& board : boards) {
    ss << board << "\n";
  }
  return ss.str();
}

bool GBTRawDataChecker::checkEvent(bool isTriggered, const std::vector<ROBoard>& regs, const std::vector<ROBoard>& locs, const InteractionRecord& ir)
{
  /// Checks the cards belonging to the same BC
  mEventDebugMsg.clear();
  if (!checkRegLocConsistency(regs, locs, ir)) {
    return false;
  }

  if (!checkConsistency(regs) || !checkConsistency(locs)) {
    return false;
  }

  if (!isTriggered) {
    if (!checkLocalBoardSize(locs)) {
      return false;
    }

    if (!checkMasks(locs)) {
      return false;
    }
  }

  return true;
}

InteractionRecord GBTRawDataChecker::getRawIR(uint8_t id, bool isTrigger, InteractionRecord ir) const
{
  /// Returns the bc as it was set by electronics (before corrections)
  if (isTrigger) {
    return ir;
  }
  auto delay = mElectronicsDelay.localToBC;
  if (id >= crateparams::sMaxNBoardsInLink) {
    delay -= mElectronicsDelay.localToReg;
  }
  applyElectronicsDelay(ir.orbit, ir.bc, -delay, mResetVal);
  return ir;
}

uint8_t GBTRawDataChecker::getElinkId(const ROBoard& board) const
{
  /// Returns the e-link ID
  if (raw::isLoc(board.statusWord)) {
    return board.boardId % 8;
  }
  return 8 + board.boardId % 8;
}

void GBTRawDataChecker::clearChecked(bool isTriggered, bool clearTrigEvents)
{
  /// Clears the checked events

  auto& boards = isTriggered ? mBoardsTrig : mBoardsSelfTrig;
  auto& lastIndexes = isTriggered ? mLastIndexTrig : mLastIndexSelfTrig;
  // Create a new board map with the checked events stripped
  for (auto& lastIdxItem : lastIndexes) {
    auto firstIdx = lastIdxItem.second + 1;
    auto& boardVec = boards[lastIdxItem.first];
    boards[lastIdxItem.first].erase(boardVec.begin(), boardVec.begin() + firstIdx);
  }

  if (clearTrigEvents) {
    // Clears the map with the processed triggers
    auto& lastCompleteTrigIR = isTriggered ? mLastCompleteIRTrig : mLastCompleteIRSelfTrig;
    auto up = mTrigEvents.upper_bound(lastCompleteTrigIR);
    mTrigEvents.erase(mTrigEvents.begin(), up);
    for (auto& busyInfoVec : mBusyPeriods) {
      auto upBusy = std::upper_bound(busyInfoVec.begin(), busyInfoVec.end(), lastCompleteTrigIR, [](const InteractionRecord& ir, const BusyInfo& busyInfo) { return ir <= busyInfo.interactionRecord; });
      if (upBusy != busyInfoVec.begin()) {
        --upBusy;
      }
      busyInfoVec.erase(busyInfoVec.begin(), upBusy);
    }
  }
}

bool GBTRawDataChecker::isCompleteSelfTrigEvent(const o2::InteractionRecord& ir) const
{
  /// Checks if the self-triggered events are complete

  // The regional board information in self-triggered events is delayed
  // compared to triggered events.
  // So, we expect information from a previous orbit after having received an orbit trigger.
  // Let us check that we have all boards with the same orbit

  bool isIncluded = false;
  for (uint8_t ireg = 8; ireg < crateparams::sNELinksPerGBT; ++ireg) {
    auto& boards = mBoardsSelfTrig[ireg];
    if (!boards.empty()) {
      if (boards.back().interactionRecord.orbit == ir.orbit) {
        return false;
      }
      if (boards.front().interactionRecord.orbit <= ir.orbit) {
        isIncluded = true;
      }
    }
  }
  return isIncluded;
}

unsigned int GBTRawDataChecker::getLastCompleteTrigEvent()
{
  /// Checks if we have a triggered event with the information from all active boards
  /// The function returns true if it finds a complete event
  /// and it returns its interaction record as well.

  // The information for an event comes at different times for different boards,
  // depending on the length of the self-triggered event for that board.
  // So, before testing the consistency of the event,
  // we must wait to have received the information from all boards.
  // This can be checked in triggered events, since all boards should be present.

  unsigned int completeMask = 0;

  // Check if we have a triggered event with the information from all active boards
  mLastCompleteIRSelfTrig = o2::InteractionRecord();
  uint16_t fullMask = (3 << 8) | mCrateMask;
  auto trigEventIt = mTrigEvents.rbegin();
  auto end = mTrigEvents.rend();
  for (; trigEventIt != end; ++trigEventIt) {
    if ((trigEventIt->second & fullMask) == fullMask) {
      // The trigger events contain the unprocessed events for both triggered and self-triggered events
      // These might not be synchronized (typically the latest complete self-triggered events lie behind)
      // If the latest IR in memory is more recent than the current complete event found,
      // then it means that we need to wait for more HBs.
      if (mLastCompleteIRTrig.isDummy() || mLastCompleteIRTrig < trigEventIt->first) {
        completeMask |= 1;
        mLastCompleteIRTrig = trigEventIt->first;
      }
      auto trIt = trigEventIt;
      while (trIt != end) {
        if (isCompleteSelfTrigEvent(trIt->first)) {
          completeMask |= (1 << 1);
          mLastCompleteIRSelfTrig = trIt->first;
          break;
        }
        ++trIt;
      }

      return completeMask;
    }
  }

  return completeMask;
}

bool GBTRawDataChecker::runCheckEvents(unsigned int completeMask)
{
  /// Runs the checker if needed

  bool isOk = true;

  if (completeMask & 0x1) {
    sortEvents(true);
    isOk &= checkEvents(true);
    // This is needed to clear vectors in runs with no self-triggered events
    bool clearTrigger = true;
    for (auto infos : mBoardsSelfTrig) {
      if (!infos.empty()) {
        clearTrigger = false;
        break;
      }
    }
    clearChecked(true, clearTrigger);
  }

  if (completeMask & 0x2) {
    sortEvents(false);
    isOk &= checkEvents(false);
    clearChecked(false, true);
  }

  return isOk;
}

void GBTRawDataChecker::sortEvents(bool isTriggered)
{
  /// Sorts the event in time
  auto& orderedIndexes = isTriggered ? mOrderedIndexesTrig : mOrderedIndexesSelfTrig;
  auto& lastIndexes = isTriggered ? mLastIndexTrig : mLastIndexSelfTrig;
  auto& boards = isTriggered ? mBoardsTrig : mBoardsSelfTrig;
  auto& lastCompleteTrigEventIR = isTriggered ? mLastCompleteIRTrig : mLastCompleteIRSelfTrig;
  orderedIndexes.clear();
  lastIndexes.clear();
  for (uint8_t ilink = 0; ilink < crateparams::sNELinksPerGBT; ++ilink) {
    long int lastIdx = -1;
    for (auto boardIt = boards[ilink].begin(), end = boards[ilink].end(); boardIt != end; ++boardIt) {
      if (boardIt->interactionRecord > lastCompleteTrigEventIR) {
        break;
      }
      lastIdx = std::distance(boards[ilink].begin(), boardIt);
      orderedIndexes[boardIt->interactionRecord.toLong()].emplace_back(ilink, lastIdx);
    }
    lastIndexes[ilink] = lastIdx;
  }
}

bool GBTRawDataChecker::checkEvents(bool isTriggered)
{
  /// Checks the events
  bool isOk = true;
  auto& boards = isTriggered ? mBoardsTrig : mBoardsSelfTrig;
  auto& orderedIndexes = isTriggered ? mOrderedIndexesTrig : mOrderedIndexesSelfTrig;
  // Loop on the event indexes
  o2::InteractionRecord ir;
  for (auto& evtIdxItem : orderedIndexes) {
    // All of these boards have the same timestamp
    GBT gbtEvent;
    for (auto& evtPair : evtIdxItem.second) {
      auto& boardInfo = boards[evtPair.first][evtPair.second];

      if (raw::isLoc(boardInfo.board.statusWord)) {
        gbtEvent.locs.push_back(boardInfo.board);
      } else {
        gbtEvent.regs.push_back(boardInfo.board);
      }
      if (boardInfo.page >= 0) {
        if (std::find(gbtEvent.pages.begin(), gbtEvent.pages.end(), boardInfo.page) == gbtEvent.pages.end()) {
          gbtEvent.pages.push_back(boardInfo.page);
        }
      }
    }
    ++mStatistics[0];
    ir.setFromLong(evtIdxItem.first);
    if (!checkEvent(isTriggered, gbtEvent.regs, gbtEvent.locs, ir)) {
      std::stringstream ss;
      ss << fmt::format("BCid: 0x{:x} Orbit: 0x{:x}", ir.bc, ir.orbit);
      if (!gbtEvent.pages.empty()) {
        ss << "   [in";
        for (auto& page : gbtEvent.pages) {
          ss << std::dec << "  page: " << page << "  (line: " << 512 * page + 1 << ")  ";
        }
        ss << "]";
      }
      ss << "\n";
      isOk = false;
      ss << mEventDebugMsg << "\n";
      mDebugMsg += ss.str();
      ++mStatistics[1];
    }
  }

  return isOk;
}

bool GBTRawDataChecker::process(gsl::span<const ROBoard> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords)
{
  /// Checks the raw data
  mDebugMsg.clear();

  // Fill board information
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    if (rofIt->interactionRecord.orbit == 0xffffffff) {
      // Protection for event with orbit 0
      continue;
    }
    for (auto locIt = localBoards.begin() + rofIt->firstEntry; locIt != localBoards.begin() + rofIt->firstEntry + rofIt->nEntries; ++locIt) {
      // Find what page this event corresponds to.
      // This is useful for debugging.
      long int page = -1;
      for (auto& rofPage : pageRecords) {
        if (rofIt->firstEntry >= rofPage.firstEntry && rofIt->firstEntry < rofPage.firstEntry + rofPage.nEntries) {
          page = rofPage.interactionRecord.orbit;
          break;
        }
      }

      // Store the information per local board.
      // The information should be already ordered in time
      auto id = getElinkId(*locIt);
      auto& elinkVec = (locIt->triggerWord == 0) ? mBoardsSelfTrig[id] : mBoardsTrig[id];
      elinkVec.push_back({*locIt, rofIt->interactionRecord, page});

      auto& busyVec = mBusyPeriods[id];
      bool wasBusy = !busyVec.empty() && busyVec.back().isBusy;
      bool isBusy = locIt->statusWord & raw::sREJECTING;
      if (isBusy != wasBusy) {
        if (isBusy) {
          ++mStatistics[2];
        }
        busyVec.push_back({isBusy, getRawIR(id, locIt->triggerWord != 0, rofIt->interactionRecord)});
      }

      if (locIt->triggerWord == 0) {
        continue;
      }

      // Keep track of the trigger chosen for synchronisation
      if (locIt->triggerWord & mSyncTrigger) {
        mTrigEvents[rofIt->interactionRecord] |= (1 << id);
      }
      if (locIt->triggerWord & raw::sORB) {
        mResetVal = rofIt->interactionRecord.bc + 1;
      }

      // Compute the masks
      if (locIt->triggerWord & raw::sSOX) {
        if (raw::isLoc(locIt->statusWord)) {
          auto maskItem = mMasks.find(locIt->boardId);
          // Check if we have already a mask for this
          if (maskItem == mMasks.end()) {
            // If not, read the map
            auto& mask = mMasks[locIt->boardId];
            for (int ich = 0; ich < 4; ++ich) {
              mask.patternsBP[ich] = locIt->patternsBP[ich];
              mask.patternsNBP[ich] = locIt->patternsNBP[ich];
            }
          }
        }
      }
    } // loop on local boards
  }   // loop on ROF records

  return runCheckEvents(getLastCompleteTrigEvent());
}

void GBTRawDataChecker::clear(bool all)
{
  /// Resets the masks and flags
  if (all) {
    mMasks.clear();
  }
  mStatistics.fill(0);
}

} // namespace mid
} // namespace o2
