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

/// \file   MIDBase/ColumnDataHandler.h
/// \brief  MID digits handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   25 February 2022
#ifndef O2_MID_COLUMNDATAHANDLER_H
#define O2_MID_COLUMNDATAHANDLER_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <gsl/span>
#include "DataFormatsMID/ColumnData.h"

namespace o2
{
namespace mid
{

class ColumnDataHandler
{
 public:
  /// @brief Add single strip
  /// @param deId Detection element ID
  /// @param columnId Column ID
  /// @param lineId Local board line in the column
  /// @param strip Strip number
  /// @param cathode Anode or cathode
  void add(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode);

  /// @brief Merges digit
  /// @param col input digit
  /// @param idx index of col in the vector. Useful to keep track of what digit was merged
  /// @returns true if this is the first added digit
  bool merge(const ColumnData& col, size_t idx = 0);

  /// @brief Merges digits
  /// @param colVec span of column data
  /// @param idx Offset index of the first ColumnData in the vector. It can be different from 0 if we are merging a subspan.
  void merge(gsl::span<const ColumnData> colVec, size_t idx = 0);

  /// @brief Clears inner maps
  void clear();

  /// @brief Returns the merged data
  /// @return Vector of merged ColumnData
  std::vector<ColumnData> getMerged() const;

  /// @brief Returns the indexes of the ColumnData merged into this one
  /// @param col Merged ColumnData
  /// @return Vector of the indexes of the ColumnData merged into this one
  std::vector<size_t> getMergedIndexes(const ColumnData& col) const;

 private:
  std::unordered_map<uint16_t, ColumnData> mData{};                  /// Map of ColumnData
  std::unordered_map<uint16_t, std::vector<size_t>> mCorrespondence; /// Correspondence between input and merged ColumnData
};

} // namespace mid
} // namespace o2

#endif /* O2_MID_COLUMNDATAHANDLER_H */
