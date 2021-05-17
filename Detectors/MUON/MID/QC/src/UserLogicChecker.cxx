// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/QC/src/UserLogicChecker.cxx
/// \brief  Class to check the CRU user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 November 2020

#include "MIDQC/UserLogicChecker.h"

#include <sstream>
#include <fmt/format.h>

namespace o2
{
namespace mid
{

bool UserLogicChecker::isSame(const o2::mid::ROBoard& loc1, const o2::mid::ROBoard& loc2) const
{
  /// Tests if boards are sames
  if (loc1.statusWord == loc2.statusWord && loc1.triggerWord == loc2.triggerWord && loc1.firedChambers == loc2.firedChambers && loc1.boardId == loc2.boardId) {
    for (int ich = 0; ich < 4; ++ich) {
      if (loc1.patternsBP[ich] != loc2.patternsBP[ich] || loc1.patternsNBP[ich] != loc2.patternsNBP[ich]) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::string UserLogicChecker::printIRHex(const o2::InteractionRecord& ir) const
{
  /// Properly format interaction record
  return fmt::format("BCid: 0x{:x} Orbit: 0x{:x}", ir.bc, ir.orbit);
}

uint32_t UserLogicChecker::getId(const ROBoard& board) const
{
  /// Gets the unique ID for internal usage
  uint32_t id = static_cast<uint32_t>(board.boardId);
  if (!raw::isLoc(board.statusWord)) {
    id |= (1 << 16);
  }
  return id;
}

void UserLogicChecker::fillBoards(gsl::span<const ROBoard> data, gsl::span<const ROFRecord> rofRecords, bool isUL)
{
  /// Fills the inner structure for checks

  auto& boards = isUL ? mBoardsUL : mBoardsBare;

  // The UL rejects the events outside the SOX/EOX
  // So we should do the same
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    auto& loc = data[rofIt->firstEntry];
    auto id = getId(loc);
    auto isInside = mInsideDataTaking.find(id);
    if (isInside == mInsideDataTaking.end()) {
      mInsideDataTaking[id] = false;
      isInside = mInsideDataTaking.find(id);
    }
    if (loc.triggerWord & raw::sSOX) {
      isInside->second = true;
    } else if (loc.triggerWord & raw::sEOX) {
      isInside->second = false;
    }
    if (isInside->second) {
      boards[id].push_back({rofIt->interactionRecord, loc});
    }
  }
}

void UserLogicChecker::clearBoards()
{
  /// Clears the processed boards

  for (int itype = 0; itype < 2; ++itype) {
    auto& boards = (itype == 0) ? mBoardsUL : mBoardsBare;
    auto& lastChecked = (itype == 0) ? mLastCheckedUL : mLastCheckedBare;
    std::unordered_map<uint32_t, std::vector<boardInfo>> newBoards;
    for (auto& boardItem : boards) {
      auto lastIdxItem = lastChecked.find(boardItem.first);
      if (lastIdxItem->second != boardItem.second.size()) {
        auto& vec = newBoards[boardItem.first];
        vec.insert(vec.end(), boardItem.second.begin() + lastIdxItem->second, boardItem.second.end());
      }
    }
    boards.swap(newBoards);
    lastChecked.clear();
  }
}

bool UserLogicChecker::checkBoards(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs)
{
  /// Compares the UL output with the corresponding bare output per board
  clearBoards();
  fillBoards(bareData, bareRofs, false);
  fillBoards(ulData, ulRofs, true);
  bool isOk = true;

  for (auto& bareItem : mBoardsBare) {
    std::string boardType = (bareItem.first < 0x10000) ? "Loc" : "Reg";
    auto& lastCheckedBare = mLastCheckedBare[bareItem.first];
    auto& stats = mStatistics[bareItem.first];
    stats[0] += bareItem.second.size();
    std::stringstream ss;
    ss << "\n-----------" << std::endl;
    uint16_t boardId = (bareItem.first & 0xFFFF);
    ss << "Checking crate: " << static_cast<int>(raw::getCrateId(boardId))
       << "  " << boardType << " board: " << static_cast<int>(raw::getLocId(boardId)) << std::endl;
    auto ulItem = mBoardsUL.find(bareItem.first);
    if (ulItem == mBoardsUL.end()) {
      ss << "  cannot find " << printIRHex(bareItem.second.front().interactionRecord) << " in ul" << std::endl;
      isOk = false;
      ss << "-----------" << std::endl;
      mDebugMsg += ss.str();
      stats[1] += bareItem.second.size();
      continue;
    }
    auto& lastCheckedUL = mLastCheckedUL[ulItem->first];
    lastCheckedUL = 0;
    auto lastOk = lastCheckedUL;
    bool isCurrentOk = true;
    for (lastCheckedBare = 0; lastCheckedBare < bareItem.second.size(); ++lastCheckedBare) {
      if (lastCheckedUL == ulItem->second.size()) {
        break;
      } else if (!isSame(ulItem->second[lastCheckedUL].board, bareItem.second[lastCheckedBare].board) || ulItem->second[lastCheckedUL].interactionRecord != bareItem.second[lastCheckedBare].interactionRecord) {
        ss << "\nFirst divergence at element " << lastCheckedBare + 1 << " / " << bareItem.second.size() << ":" << std::endl;
        ss << "bare: " << printIRHex(bareItem.second[lastCheckedBare].interactionRecord) << std::endl;
        ss << bareItem.second[lastCheckedBare].board << std::endl;
        ss << "ul: " << printIRHex(ulItem->second[lastCheckedUL].interactionRecord) << std::endl;
        ss << ulItem->second[lastCheckedUL].board << std::endl;
        isCurrentOk = false;
      }
      if (!isCurrentOk) {
        if (lastOk != lastCheckedUL) {
          ss << "lastOk: " << printIRHex(ulItem->second[lastOk].interactionRecord) << std::endl;
          ss << ulItem->second[lastOk].board << std::endl;
        } else {
          ss << "lastOk: none. This is the first event!" << std::endl;
        }
        ss << "-----------" << std::endl;
        mDebugMsg += ss.str();
        stats[1] += lastCheckedUL;
        lastCheckedUL = ulItem->second.size();
        lastCheckedBare = bareItem.second.size();
        isOk = false;
        break;
      }
      lastOk = lastCheckedUL;
      ++lastCheckedUL;
    } // loop on bare data for this board ID
  }   // loop on board IDs
  return isOk;
}

std::unordered_map<uint64_t, std::vector<size_t>> UserLogicChecker::getOrderedIndexes(gsl::span<const ROFRecord> rofRecords) const
{
  // Orders data according to their IR
  std::unordered_map<uint64_t, std::vector<size_t>> orderIndexes;
  for (auto rofIt = rofRecords.begin(); rofIt != rofRecords.end(); ++rofIt) {
    // Fill the map with ordered events
    orderIndexes[rofIt->interactionRecord.toLong()].emplace_back(rofIt->firstEntry);
  }
  return orderIndexes;
}

bool UserLogicChecker::checkAll(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs)
{
  auto bareIndexes = getOrderedIndexes(bareRofs);
  auto ulIndexes = getOrderedIndexes(ulRofs);

  bool isOk = true;
  std::stringstream ss;
  InteractionRecord ir;
  for (auto& bareItem : bareIndexes) {
    auto ulItem = ulIndexes.find(bareItem.first);
    ir.setFromLong(bareItem.first);
    if (ulItem == ulIndexes.end()) {
      isOk = false;
      ss << "\nCannot find: " << printIRHex(ir) << " in ul\n";
      continue;
    }
    std::vector<size_t> auxVec = ulItem->second;
    for (auto& idx1 : bareItem.second) {
      bool found = false;
      for (auto auxIt = auxVec.begin(); auxIt != auxVec.end(); ++auxIt) {
        if (isSame(bareData[idx1], ulData[*auxIt])) {
          auxVec.erase(auxIt);
          found = true;
          break;
        }
      }
      if (!found) {
        isOk = false;
        ss << "\nOnly in bare: " << printIRHex(ir) << "\n";
        ss << "  " << bareData[idx1] << "\n";
      }
    }
    for (auto& idx2 : auxVec) {
      isOk = false;
      ss << "\nOnly in ul: " << printIRHex(ir) << "\n";
      ss << "  " << ulData[idx2] << "\n";
    }
  }

  for (auto& ulItem : ulIndexes) {
    auto bareItem = bareIndexes.find(ulItem.first);
    if (bareItem == bareIndexes.end()) {
      isOk = false;
      ir.setFromLong(ulItem.first);
      ss << "\nCannot find: " << printIRHex(ir) << " in bare\n";
    }
  }
  mDebugMsg = ss.str();
  return isOk;
}

bool UserLogicChecker::process(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs, bool isFull)
{
  /// Compares the UL output with the corresponding bare output
  mDebugMsg.clear();
  return isFull ? checkAll(bareData, bareRofs, ulData, ulRofs) : checkBoards(bareData, bareRofs, ulData, ulRofs);
}

void UserLogicChecker::clear()
{
  /// Clears debug message
  mInsideDataTaking.clear();
  mStatistics.clear();
}

std::string UserLogicChecker::getSummary() const
{
  /// Gets summary message
  std::stringstream ss;
  for (auto& statItem : mStatistics) {
    std::string boardType = (statItem.first < 0x10000) ? "Loc" : "Reg";
    double badFraction = (statItem.second[0] == 0) ? 0. : static_cast<double>(statItem.second[1]) / static_cast<double>(statItem.second[0]);
    uint16_t boardId = (statItem.first & 0xFFFF);
    ss << "Crate: " << static_cast<int>(raw::getCrateId(boardId)) << "  " << boardType << " board: " << static_cast<int>(raw::getLocId(boardId)) << "  fraction of events not in sync: " << statItem.second[1] << " / " << statItem.second[0] << " = " << badFraction << std::endl;
  }
  return ss.str();
}

} // namespace mid
} // namespace o2
