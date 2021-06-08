// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDQC/UserLogicChecker.h
/// \brief  Class to check the CRU user logic
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 November 2020
#ifndef O2_MID_USERLOGICCHECKER_H
#define O2_MID_USERLOGICCHECKER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <gsl/gsl>
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"

namespace o2
{
namespace mid
{
class UserLogicChecker
{
 public:
  bool process(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs, bool isFull = false);

  /// Gets the debug message
  std::string getDebugMessage() const { return mDebugMsg; }
  std::string getSummary() const;
  void clear();

 private:
  bool checkAll(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs);
  bool checkBoards(gsl::span<const ROBoard> bareData, gsl::span<const ROFRecord> bareRofs, gsl::span<const ROBoard> ulData, gsl::span<const ROFRecord> ulRofs);
  void clearBoards();
  void fillBoards(gsl::span<const ROBoard> data, gsl::span<const ROFRecord> rofRecords, bool isUL);
  uint32_t getId(const ROBoard& board) const;

  std::unordered_map<uint64_t, std::vector<size_t>> getOrderedIndexes(gsl::span<const ROFRecord> rofRecords) const;

  bool isSame(const o2::mid::ROBoard& loc1, const o2::mid::ROBoard& loc2) const;
  std::string printIRHex(const o2::InteractionRecord& ir) const;

  std::string mDebugMsg{}; /// Debug message

  std::unordered_map<uint32_t, bool> mInsideDataTaking{}; /// Flag to assess if we are inside SOX-EOX

  std::unordered_map<uint32_t, std::array<unsigned long int, 2>> mStatistics{}; /// Board statistics

  struct boardInfo {
    InteractionRecord interactionRecord;
    ROBoard board;
  };

  std::unordered_map<uint32_t, std::vector<boardInfo>> mBoardsBare; //! Bare boards info
  std::unordered_map<uint32_t, std::vector<boardInfo>> mBoardsUL;   //! UL Boards info
  std::unordered_map<uint32_t, size_t> mLastCheckedBare;            //! Last checked bare
  std::unordered_map<uint32_t, size_t> mLastCheckedUL;              //! Last checked UL
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_USERLOGICCHECKER_H */
