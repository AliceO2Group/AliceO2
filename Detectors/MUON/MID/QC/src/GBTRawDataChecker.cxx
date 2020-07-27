// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

void GBTRawDataChecker::init(uint16_t feeId, uint8_t mask)
{
  /// Initializer
  mFeeId = feeId;
  mCrateMask = mask;
}

bool GBTRawDataChecker::checkLocalBoardSize(const LocalBoardRO& board)
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

bool GBTRawDataChecker::checkLocalBoardSize(const std::vector<LocalBoardRO>& boards)
{
  /// Checks that the boards have the expected non-null patterns
  for (auto& board : boards) {
    if (!checkLocalBoardSize(board)) {
      return false;
    }
  }
  return true;
}

bool GBTRawDataChecker::checkConsistency(const LocalBoardRO& board)
{
  /// Checks that the event information is consistent

  bool isSoxOrReset = board.triggerWord & (raw::sSOX | raw::sEOX | raw::sRESET);
  bool isCalib = raw::isCalibration(board.triggerWord);
  bool isPhys = board.triggerWord & raw::sPHY;

  if (isPhys) {
    if (isCalib) {
      mEventDebugMsg += "inconsistent trigger: calibration and physics trigger cannot be fired together\n";
      return false;
    }
    if (raw::isLoc(board.statusWord)) {
      if (board.firedChambers) {
        mEventDebugMsg += "inconsistent trigger: fired chambers should be 0\n";
        return false;
      }
    }
  }
  if (isSoxOrReset && (isCalib || isPhys)) {
    mEventDebugMsg += "inconsistent trigger: cannot be SOX and calibration\n";
    return false;
  }

  return true;
}

bool GBTRawDataChecker::checkConsistency(const std::vector<LocalBoardRO>& boards)
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

bool GBTRawDataChecker::checkMasks(const std::vector<LocalBoardRO>& locs)
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

bool GBTRawDataChecker::checkRegLocConsistency(const std::vector<LocalBoardRO>& regs, const std::vector<LocalBoardRO>& locs)
{
  /// Checks consistency between local and regional info
  uint8_t regFired{0};
  for (auto& reg : regs) {
    uint8_t ireg = crateparams::getLocId(reg.boardId) % 2;
    auto busyItem = mBusyFlagSelfTrig.find(8 + ireg);
    if (reg.triggerWord == 0) {
      // Self-triggered event: check the decision
      regFired |= (reg.firedChambers << (4 * ireg));
    } else {
      // Triggered event: all active boards must answer
      regFired |= (mCrateMask & (0xF << (4 * ireg)));
    }
  }
  uint8_t locFired{0}, locBusy{0};
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
    }
  }

  // The XOR returns 1 in case of a difference
  uint8_t problems = (regFired ^ locFired);

  if (problems) {
    // It can be that a busy signal was raised by one of the board in previous events
    // If the board is still busy it will not answer.
    uint8_t busy{0};
    for (uint8_t iboard = 0; iboard < crateparams::sNELinksPerGBT; ++iboard) {
      auto busyItem = mBusyFlagSelfTrig.find(iboard);
      if (busyItem != mBusyFlagSelfTrig.end() && busyItem->second) {
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

std::string GBTRawDataChecker::printBoards(const std::vector<LocalBoardRO>& boards) const
{
  /// Prints the boards
  std::stringstream ss;
  for (auto& board : boards) {
    ss << board << "\n";
  }
  return ss.str();
}

bool GBTRawDataChecker::checkEvent(bool isTriggered, const std::vector<LocalBoardRO>& regs, const std::vector<LocalBoardRO>& locs)
{
  /// Checks the cards belonging to the same BC
  mEventDebugMsg.clear();
  if (!checkRegLocConsistency(regs, locs)) {
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

uint8_t GBTRawDataChecker::getElinkId(const LocalBoardRO& board) const
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
  std::unordered_map<uint8_t, std::vector<BoardInfo>> newBoards{};
  for (auto& lastIdxItem : lastIndexes) {
    auto firstIdx = lastIdxItem.second + 1;
    auto& boardVec = boards[lastIdxItem.first];
    if (firstIdx < boardVec.size()) {
      auto& newVec = newBoards[lastIdxItem.first];
      newVec.insert(newVec.end(), boardVec.begin() + firstIdx, boardVec.end());
    }
  }
  boards.swap(newBoards);

  if (clearTrigEvents) {
    // Clears the map with the processed triggers
    auto& lastCompleteTrigIR = isTriggered ? mLastCompleteIRTrig : mLastCompleteIRSelfTrig;
    auto low = mTrigEvents.begin();
    auto up = mTrigEvents.upper_bound(lastCompleteTrigIR);
    mTrigEvents.erase(low, up);
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
  for (uint8_t ireg = 8; ireg < 10; ++ireg) {
    auto item = mBoardsSelfTrig.find(ireg);
    if (item != mBoardsSelfTrig.end()) {
      if (item->second.front().interactionRecord.orbit <= ir.orbit) {
        isIncluded = true;
      }
      if (item->second.back().interactionRecord.orbit <= ir.orbit) {
        return false;
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
      // These might not be synchronized (typically the latest complete self-triggered events lies behind)
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

void GBTRawDataChecker::sortEvents(bool isTriggered)
{
  /// Sorts the event in time
  auto& orderedIndexes = isTriggered ? mOrderedIndexesTrig : mOrderedIndexesSelfTrig;
  auto& lastIndexes = isTriggered ? mLastIndexTrig : mLastIndexSelfTrig;
  auto& boards = isTriggered ? mBoardsTrig : mBoardsSelfTrig;
  auto& lastCompleteTrigEventIR = isTriggered ? mLastCompleteIRTrig : mLastCompleteIRSelfTrig;
  orderedIndexes.clear();
  lastIndexes.clear();
  for (auto& boardItem : boards) {
    size_t lastIdx = 0;
    for (auto boardIt = boardItem.second.begin(), end = boardItem.second.end(); boardIt != end; ++boardIt) {
      if (boardIt->interactionRecord > lastCompleteTrigEventIR) {
        break;
      }
      lastIdx = std::distance(boardItem.second.begin(), boardIt);
      orderedIndexes[boardIt->interactionRecord].emplace_back(boardItem.first, lastIdx);
    }
    lastIndexes[boardItem.first] = lastIdx;
  }
}

bool GBTRawDataChecker::checkEvents(bool isTriggered)
{
  /// Checks the events
  bool isOk = true;
  auto& boards = isTriggered ? mBoardsTrig : mBoardsSelfTrig;
  auto& orderedIndexes = isTriggered ? mOrderedIndexesTrig : mOrderedIndexesSelfTrig;
  auto& busyFlag = isTriggered ? mBusyFlagTrig : mBusyFlagSelfTrig;
  // Loop on the event indexes
  for (auto& evtIdxItem : orderedIndexes) {
    // All of these boards have the same timestamp
    GBT gbtEvent;
    bool busyRaised = false;
    for (auto& evtPair : evtIdxItem.second) {
      auto& boardInfo = boards[evtPair.first][evtPair.second];
      uint8_t triggerId = boardInfo.board.triggerWord;
      auto elinkId = getElinkId(boardInfo.board);

      bool isBusy = ((boardInfo.board.statusWord & raw::sREJECTING) != 0);
      busyRaised |= isBusy;
      busyFlag[elinkId] = isBusy;
      if (isBusy && !isTriggered) {
        // This is a special event that just signals a busy.
        // Do not add the board to the events to be tested.
        // Even because this event can have the same IR and triggerWord (0) of a self-triggered event
        continue;
      }
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
    if (busyRaised && !isTriggered) {
      ++mStatistics[2];
    }
    ++mStatistics[0];
    if (!checkEvent(isTriggered, gbtEvent.regs, gbtEvent.locs)) {
      std::stringstream ss;
      ss << fmt::format("BCid: 0x{:x} Orbit: 0x{:x}", evtIdxItem.first.bc, evtIdxItem.first.orbit);
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

bool GBTRawDataChecker::process(gsl::span<const LocalBoardRO> localBoards, gsl::span<const ROFRecord> rofRecords, gsl::span<const ROFRecord> pageRecords)
{
  /// Checks the raw data
  mDebugMsg.clear();

  // Fill board information
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
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

      if (locIt->triggerWord == 0) {
        continue;
      }

      // Keep track of the busy
      if (locIt->statusWord & raw::sREJECTING) {
        auto& selfVec = mBoardsSelfTrig[id];
        auto board = *locIt;
        board.triggerWord = 0;
        auto ir = rofIt->interactionRecord;
        if (id >= crateparams::sMaxNBoardsInLink) {
          uint16_t delayRegLocal = mElectronicsDelay.regToLocal;
          if (rofIt->interactionRecord.bc < delayRegLocal) {
            ir -= (constants::lhc::LHCMaxBunches - mResetVal - 1);
          }
          ir -= delayRegLocal;
        }
        selfVec.push_back({*locIt, ir, page});
      }

      // Keep track of the orbit triggers
      if (locIt->triggerWord & raw::sORB) {
        mTrigEvents[rofIt->interactionRecord] |= (1 << id);
        mResetVal = rofIt->interactionRecord.bc;
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

  auto completeMask = getLastCompleteTrigEvent();

  bool isOk = true;

  if (completeMask & 0x1) {
    sortEvents(true);
    isOk &= checkEvents(true);
    clearChecked(true, mBoardsSelfTrig.empty());
  }

  if (completeMask & 0x2) {
    sortEvents(false);
    isOk &= checkEvents(false);
    clearChecked(false, true);
  }

  return isOk;
}

void GBTRawDataChecker::clear()
{
  /// Resets the masks and flags
  mMasks.clear();
  mBusyFlagTrig.clear();
  mBusyFlagSelfTrig.clear();
  mStatistics.fill(0);
}

} // namespace mid
} // namespace o2
