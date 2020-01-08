// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   CRUBareDataChecker.h
/// \brief  Class to check the bare data from the CRU
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   9 December 2019
#ifndef O2_MID_CRUBAREDATACHECKER_H
#define O2_MID_CRUBAREDATACHECKER_H

#include <cstdint>
#include <vector>
#include <string>
#include <map>
#include <gsl/gsl>
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/LocalBoardRO.h"

namespace o2
{
namespace mid
{
class CRUBareDataChecker
{
 public:
  bool process(gsl::span<const LocalBoardRO> localBoards, gsl::span<const ROFRecord> rofRecords, bool resetStat = true);
  /// Gets the number of processed events
  unsigned int getNBCsProcessed() const { return mStatistics[0]; }
  /// Gets the number of faulty events
  unsigned int getNBCsFaulty() const { return mStatistics[1]; }
  /// Gets the
  std::string getDebugMessage() const { return mDebugMsg; }

 private:
  bool checkBC(const std::vector<LocalBoardRO>& regs, const std::vector<LocalBoardRO>& locs, std::string& debugMsg);
  bool checkSameEventWord(const std::vector<LocalBoardRO>& boards, uint8_t refEventWord) const;
  bool checkConsistency(const LocalBoardRO& board) const;
  bool checkConsistency(const std::vector<LocalBoardRO>& boards) const;
  bool checkPatterns(const LocalBoardRO& board, uint8_t expected = 0xFF) const;
  bool checkPatterns(const std::vector<LocalBoardRO>& boards, uint8_t expected = 0xFF) const;

  std::map<uint64_t, std::vector<size_t>> mOrderIndexes; /// Map for time ordering the entries
  std::string mDebugMsg{};                               /// Debug message
  std::array<unsigned long int, 2> mStatistics{};        /// Processed events statistics
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CRUBAREDATACHECKER_H */
