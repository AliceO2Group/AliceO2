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

/// \file   MID/Base/src/ColumnDataHandler.cxx
/// \brief  MID digits handler
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   25 February 2022

#include "MIDBase/ColumnDataHandler.h"

namespace o2
{
namespace mid
{

void ColumnDataHandler::add(uint8_t deId, uint8_t columnId, int lineId, int strip, int cathode)
{
  auto uniqueId = getColumnDataUniqueId(deId, columnId);
  auto dataIt = mData.find(uniqueId);
  if (dataIt == mData.end()) {
    auto& newCol = mData[uniqueId];
    newCol.deId = deId;
    newCol.columnId = columnId;
    dataIt = mData.find(uniqueId);
  }
  dataIt->second.addStrip(strip, cathode, lineId);
}

bool ColumnDataHandler::merge(const ColumnData& col, size_t idx)
{
  auto uniqueId = getColumnDataUniqueId(col.deId, col.columnId);
  auto dataIt = mData.find(uniqueId);
  bool isNew = false;
  if (dataIt == mData.end()) {
    isNew = true;
    mData[uniqueId] = col;
  } else {
    dataIt->second |= col;
  }
  mCorrespondence[uniqueId].emplace_back(idx);

  return isNew;
}

void ColumnDataHandler::merge(gsl::span<const ColumnData> colVec, size_t idx)
{
  for (size_t ic = 0, end = colVec.size(); ic < end; ++ic) {
    merge(colVec[ic], idx + ic);
  }
}

std::vector<ColumnData> ColumnDataHandler::getMerged() const
{
  /// Gets the masks
  std::vector<ColumnData> data;
  for (auto& item : mData) {
    data.emplace_back(item.second);
  }
  return data;
}

void ColumnDataHandler::clear()
{
  mData.clear();
  mCorrespondence.clear();
}

std::vector<size_t> ColumnDataHandler::getMergedIndexes(const ColumnData& col) const
{
  auto uniqueId = getColumnDataUniqueId(col.deId, col.columnId);
  auto corrIt = mCorrespondence.find(uniqueId);
  if (corrIt == mCorrespondence.end()) {
    return std::vector<size_t>{};
  }
  return corrIt->second;
}

} // namespace mid
} // namespace o2
