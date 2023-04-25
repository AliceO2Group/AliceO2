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

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#endif

#include "ITStracking/Constants.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

template <unsigned char maxRoadSize = 5>
class Road final
{
 public:
  GPUhd() Road() : mCellIds{}, mRoadSize{}, mIsFakeRoad{} { resetRoad(); }
  GPUhd() Road(int cellLayer, int cellId) : Road() { addCell(cellLayer, cellId); }

  int getRoadSize() const;
  int getLabel() const;
  void setLabel(const int);
  bool isFakeRoad() const;
  void setFakeRoad(const bool);
  GPUhdni() int& operator[](const int&);

  GPUhd() void resetRoad()
  {
    for (int i = 0; i < maxRoadSize; i++) {
      mCellIds[i] = constants::its::UnusedIndex;
    }
    mRoadSize = 0;
  }

  GPUhd() void addCell(int cellLayer, int cellId)
  {
    if (mCellIds[cellLayer] == constants::its::UnusedIndex) {
      ++mRoadSize;
    }

    mCellIds[cellLayer] = cellId;
  }

 private:
  int mCellIds[maxRoadSize];
  // int mLabel;
  unsigned char mRoadSize;
  bool mIsFakeRoad;
};

template <unsigned char maxRoadSize>
inline int Road<maxRoadSize>::getRoadSize() const
{
  return mRoadSize;
}

// template <unsigned char maxRoadSize>
// inline int Road<maxRoadSize>::getLabel() const
// {
//   return mLabel;
// }

// template <unsigned char maxRoadSize>
// inline void Road<maxRoadSize>::setLabel(const int label)
// {
//   mLabel = label;
// }

template <unsigned char maxRoadSize>
GPUhdi() int& Road<maxRoadSize>::operator[](const int& i)
{
  return mCellIds[i];
}

template <unsigned char maxRoadSize>
inline bool Road<maxRoadSize>::isFakeRoad() const
{
  return mIsFakeRoad;
}

template <unsigned char maxRoadSize>
inline void Road<maxRoadSize>::setFakeRoad(const bool isFakeRoad)
{
  mIsFakeRoad = isFakeRoad;
}
} // namespace its
} // namespace o2

#endif