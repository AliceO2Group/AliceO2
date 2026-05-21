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
/// \file Cluster.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CACLUSTER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTER_H_

#include <array>
#include "ITStracking/Constants.h"
#include "ITStracking/MathUtils.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"

namespace o2::its
{

template <int>
class IndexTableUtils;

struct Cluster final {
  GPUhdDefault() Cluster() = default;
  GPUhd() Cluster(const float x, const float y, const float z, const int idx)
    : xCoordinate(x),
      yCoordinate(y),
      zCoordinate(z),
      phi(math_utils::computeNormalizedPhi(x, y)),
      radius(math_utils::hypot(x, y)),
      clusterId(idx),
      indexTableBinIndex(0) {}
  GPUhd() void print() const;

  float xCoordinate{constants::UnsetValue};
  float yCoordinate{constants::UnsetValue};
  float zCoordinate{constants::UnsetValue};
  float phi{constants::UnsetValue};
  float radius{constants::UnsetValue};
  int clusterId{constants::UnusedIndex};
  int indexTableBinIndex{constants::UnusedIndex};

  ClassDefNV(Cluster, 1);
};

struct TrackingFrameInfo final {
  GPUhdDefault() TrackingFrameInfo() = default;
  GPUhd() TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, const std::array<float, 2>& posTF, const std::array<float, 3>& covTF)
    : xCoordinate(x), yCoordinate(y), zCoordinate(z), xTrackingFrame(xTF), alphaTrackingFrame(alpha), positionTrackingFrame(posTF), covarianceTrackingFrame(covTF) {}

  GPUhd() void print() const;

  float xCoordinate{constants::UnsetValue};
  float yCoordinate{constants::UnsetValue};
  float zCoordinate{constants::UnsetValue};
  float xTrackingFrame{constants::UnsetValue};
  float alphaTrackingFrame{constants::UnsetValue};
  std::array<float, 2> positionTrackingFrame{constants::UnsetValue, constants::UnsetValue};
  std::array<float, 3> covarianceTrackingFrame{constants::UnsetValue, constants::UnsetValue, constants::UnsetValue};

  ClassDefNV(TrackingFrameInfo, 1);
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
