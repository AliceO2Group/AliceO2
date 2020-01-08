// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CRUBareDataChecker.cxx
/// \brief  Class to check the bare data from the CRU
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   9 December 2019

#include "CRUBareDataChecker.h"

#include <array>
#include <sstream>
#include "MIDRaw/CrateParameters.h"

namespace o2
{
namespace mid
{

bool CRUBareDataChecker::checkSameEventWord(const std::vector<LocalBoardRO>& boards, uint8_t refEventWord) const
{
  /// Checks the event word
  for (auto loc : boards) {
    if (loc.eventWord != refEventWord) {
      return false;
    }
  }
  return true;
}

bool CRUBareDataChecker::checkPatterns(const LocalBoardRO& board, uint8_t expected) const
{
  /// Checks that the board has the expected non-null patterns
  uint8_t inputs = (expected > 0xF) ? board.firedChambers : expected;
  for (int ich = 0; ich < 4; ++ich) {
    bool isExpectedNull = (((inputs >> ich) & 0x1) == 0);
    bool isNull = (board.patternsBP[ich] == 0 && board.patternsNBP[ich] == 0);
    if (isExpectedNull != isNull) {
      return false;
    }
  }
  return true;
}

bool CRUBareDataChecker::checkPatterns(const std::vector<LocalBoardRO>& boards, uint8_t expected) const
{
  /// Checks that the boards have the expected non-null patterns
  for (auto& board : boards) {
    if (!checkPatterns(board, expected)) {
      return false;
    }
  }
  return true;
}

bool CRUBareDataChecker::checkConsistency(const LocalBoardRO& board) const
{
  /// Checks that the event information is consistent

  bool isSoxOrReset = board.eventWord & 0xc2;
  bool isCalib = crateparams::isCalibration(board.eventWord);
  bool isPhysOrHC = board.eventWord & 0x5;

  if (isPhysOrHC) {
    if (isCalib) {
      return false;
    }
    if (crateparams::isLoc(board.statusWord)) {
      if (board.firedChambers) {
        return false;
      }
    }
  }
  if (isSoxOrReset && (isCalib || isPhysOrHC)) {
    return false;
  }

  return true;
}

bool CRUBareDataChecker::checkConsistency(const std::vector<LocalBoardRO>& boards) const
{
  /// Checks that the event information is consistent
  for (auto& board : boards) {
    if (!checkConsistency(board)) {
      return false;
    }
  }
  return true;
}

bool CRUBareDataChecker::checkBC(const std::vector<LocalBoardRO>& regs, const std::vector<LocalBoardRO>& locs, std::string& debugMsg)
{
  /// Checks the cards belonging to the same BC
  bool isOk = true;

  if (locs.size() != 4 * regs.size()) {
    std::stringstream ss;
    ss << "missing cards info:  nLocs (" << locs.size() << ") != 4 x nRegs (" << regs.size() << ");  ";
    debugMsg += ss.str();
    isOk = false;
  }

  uint8_t refEventWord = 0;
  if (!regs.empty()) {
    refEventWord = regs.front().eventWord;
  } else if (!locs.empty()) {
    // FIXME: in some files, a series of 0xeeee wrongly added before the new RDH
    // This is a known problem, so we do not check further in this case
    // if (locs.front().statusWord == 0xee && locs.front().eventWord == 0xee && locs.front().firedChambers == 0xe) {
    //   return true;
    // }
    refEventWord = locs.front().eventWord;
  }

  if (!checkSameEventWord(regs, refEventWord) || !checkSameEventWord(locs, refEventWord)) {
    debugMsg += "wrong event word;  ";
    isOk = false;
  }

  if (!checkPatterns(regs, 0) || !checkPatterns(locs)) {
    debugMsg += "wrong size;  ";
    isOk = false;
  }

  if (!checkConsistency(regs) || !checkConsistency(locs)) {
    debugMsg += "inconsistency in the event;  ";
    isOk = false;
  }

  return isOk;
}

bool CRUBareDataChecker::process(gsl::span<const LocalBoardRO> localBoards, gsl::span<const ROFRecord> rofRecords, bool resetStat)
{
  /// Checks the raw data

  bool isOk = true;
  if (resetStat) {
    mStatistics.fill(0);
  }

  mDebugMsg.clear();

  // Fill the map with ordered events
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    mOrderIndexes[rofIt->interactionRecord.toLong()].emplace_back(rofIt - rofRecords.begin());
  }

  std::vector<LocalBoardRO> locs;
  std::vector<LocalBoardRO> regs;

  for (auto& item : mOrderIndexes) {
    for (auto& idx : item.second) {
      // In principle all of these ROF records have the same timestamp
      for (size_t iloc = rofRecords[idx].firstEntry; iloc < rofRecords[idx].firstEntry + rofRecords[idx].nEntries; ++iloc) {
        if (crateparams::isLoc(localBoards[iloc].statusWord)) {
          // This is a local card
          locs.push_back(localBoards[iloc]);
        } else {
          regs.push_back(localBoards[iloc]);
        }
      }
    }
    // std::sort(locs.begin(), locs.end(), [](const LocalBoardRO& a, const LocalBoardRO& b) { return a.boardId < b.boardId; });
    ++mStatistics[0];
    std::string debugStr;
    if (!checkBC(regs, locs, debugStr)) {
      isOk = false;
      std::stringstream ss;
      ss << std::hex << std::showbase << rofRecords[item.second.front()].interactionRecord << "  problems: " << debugStr << "\n";
      for (auto& reg : regs) {
        ss << "  " << reg << "\n";
      }
      for (auto& loc : locs) {
        ss << "  " << loc << "\n";
      }
      mDebugMsg += ss.str();
      ++mStatistics[1];
    }

    locs.clear();
    regs.clear();
  }

  // Clear the inner objects when the computation is done
  mOrderIndexes.clear();

  return isOk;
}

} // namespace mid
} // namespace o2
