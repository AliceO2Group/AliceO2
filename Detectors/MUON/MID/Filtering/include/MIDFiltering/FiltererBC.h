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

/// \file   MIDFiltering/FiltererBC.h
/// \brief  BC filterer for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 January 2023

#ifndef O2_MID_FILTERERBC_H
#define O2_MID_FILTERERBC_H

#include <vector>
#include <gsl/gsl>
#include "CommonDataFormat/BunchFilling.h"
#include "DataFormatsMID/ROFRecord.h"

namespace o2
{
namespace mid
{
/// Filtering algorithm for MID
class FiltererBC
{
 public:
  /// @brief Filters the data BC
  /// @param rofRecords ROF records
  /// @returns Vector of filtered ROF records
  std::vector<ROFRecord> process(gsl::span<const ROFRecord> rofRecords);

  /// @brief Set the maximum BC diff in the lower side
  /// @param bcDiffLow maximum BC diff in the lower side (negative value)
  void setBCDiffLow(int bcDiffLow) { mBCDiffLow = bcDiffLow; }

  /// @brief Set the maximum BC diff in the upper side
  /// @param bcDiffHigh maximum BC diff in the upper side (positive value)
  void setBCDiffHigh(int bcDiffHigh) { mBCDiffHigh = bcDiffHigh; }

  /// @brief Only selects BCs but do not merge them
  /// @param selectOnly Flag to only select BCs without merging events
  void setSelectOnly(bool selectOnly) { mSelectOnly = selectOnly; }

  /// @brief Sets the bunch filling scheme
  /// @param bunchFilling Bunch filling scheme
  void setBunchFilling(const BunchFilling& bunchFilling) { mBunchFilling = bunchFilling; }

 private:
  /// @brief Returns the matched collision BC
  /// @param bc BC to be checked
  /// @return The BC of the matched collision or -1 if there is no match
  int matchedCollision(int bc);

  int mBCDiffLow = 0;         /// Maximum BC diff on the lower side
  int mBCDiffHigh = 0;        /// Maximum BC diff on the higher side
  bool mSelectOnly = false;   /// Flag to only select BCs without merging digits
  BunchFilling mBunchFilling; /// Bunch filling scheme
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_FILTERERBC_H */
