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
///
/// \file Road.h
/// \brief
///

#ifndef TRACKINGCA_INCLUDE_ROAD_H
#define TRACKINGCA_INCLUDE_ROAD_H

#include <array>

#include "ITStracking/Constants.h"
#include "GPUCommonDef.h"

namespace o2::its
{

template <unsigned char maxRoadSize>
class Road final
{
 public:
  GPUhdDefault() Road() = default;
  GPUhd() Road(int cellLayer, int cellId) : Road() { addCell(cellLayer, cellId); }

  GPUhdDefault() Road(const Road&) = default;
  GPUhdDefault() Road(Road&&) noexcept = default;
  GPUhdDefault() ~Road() = default;

  GPUhdDefault() Road& operator=(const Road&) = default;
  GPUhdDefault() Road& operator=(Road&&) noexcept = default;

  GPUhdi() uint8_t getRoadSize() const { return mRoadSize; }
  GPUhdi() bool isFakeRoad() const { return mIsFakeRoad; }
  GPUhdi() void setFakeRoad(const bool fake) { mIsFakeRoad = fake; }
  GPUhdi() int& operator[](const int& i) { return mCellIds[i]; }
  GPUhdi() int operator[](const int& i) const { return mCellIds[i]; }

  GPUhd() void resetRoad()
  {
    for (int i = 0; i < maxRoadSize; i++) {
      mCellIds[i] = constants::UnusedIndex;
    }
    mRoadSize = 0;
  }

  GPUhd() void addCell(int cellLayer, int cellId)
  {
    if (mCellIds[cellLayer] == constants::UnusedIndex) {
      ++mRoadSize;
    }

    mCellIds[cellLayer] = cellId;
  }

 private:
  std::array<int, maxRoadSize> mCellIds = constants::helpers::initArray<int, maxRoadSize, constants::UnusedIndex>();
  unsigned char mRoadSize{0};
  bool mIsFakeRoad{false};
};

} // namespace o2::its

#endif
