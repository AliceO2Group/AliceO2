// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Road.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_ROAD_H_
#define TRACKINGITSU_INCLUDE_ROAD_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#endif

#include "ITStracking/Constants.h"
#include "GPUCommonDef.h"

namespace o2
{
namespace its
{

class Road final
{
 public:
  Road();
  Road(int, int);

  int getRoadSize() const;
  int getLabel() const;
  void setLabel(const int);
  bool isFakeRoad() const;
  void setFakeRoad(const bool);
  GPUhdni() int& operator[](const int&);

  void resetRoad();
  void addCell(int, int);

 private:
  int mCellIds[constants::its::CellsPerRoad];
  int mRoadSize;
  int mLabel;
  bool mIsFakeRoad;
};

inline int Road::getRoadSize() const { return mRoadSize; }

inline int Road::getLabel() const { return mLabel; }

inline void Road::setLabel(const int label) { mLabel = label; }

GPUhdi() int& Road::operator[](const int& i) { return mCellIds[i]; }

inline bool Road::isFakeRoad() const { return mIsFakeRoad; }

inline void Road::setFakeRoad(const bool isFakeRoad) { mIsFakeRoad = isFakeRoad; }
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_ROAD_H_ */
