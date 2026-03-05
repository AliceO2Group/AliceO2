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
#include "GPUCommonRtypes.h"
#include "GPUCommonDef.h"

namespace o2::its
{

template <int>
class IndexTableUtils;

struct Cluster final {
  GPUhdDefault() Cluster() = default;
  GPUhd() Cluster(const float x, const float y, const float z, const int idx);
  template <int nLayers>
  GPUhd() Cluster(const int, const IndexTableUtils<nLayers>& utils, const Cluster&);
  template <int nLayers>
  GPUhd() Cluster(const int, const float3&, const IndexTableUtils<nLayers>& utils, const Cluster&);
  GPUhdDefault() Cluster(const Cluster&) = default;
  GPUhdDefault() Cluster(Cluster&&) noexcept = default;
  GPUhdDefault() ~Cluster() = default;

  GPUhdDefault() Cluster& operator=(const Cluster&) = default;
  GPUhdDefault() Cluster& operator=(Cluster&&) noexcept = default;
  GPUhdDefault() bool operator==(const Cluster&) const = default;

  GPUhd() void print() const;

  float xCoordinate{-999.f};
  float yCoordinate{-999.f};
  float zCoordinate{-999.f};
  float phi{-999.f};
  float radius{-999.f};
  int clusterId{constants::UnusedIndex};
  int indexTableBinIndex{constants::UnusedIndex};

  ClassDefNV(Cluster, 1);
};

struct TrackingFrameInfo final {
  GPUhdDefault() TrackingFrameInfo() = default;
  GPUhd() TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, std::array<float, 2>&& posTF, std::array<float, 3>&& covTF);
  GPUhdDefault() TrackingFrameInfo(const TrackingFrameInfo&) = default;
  GPUhdDefault() TrackingFrameInfo(TrackingFrameInfo&&) noexcept = default;
  GPUhdDefault() ~TrackingFrameInfo() = default;

  GPUhdDefault() TrackingFrameInfo& operator=(const TrackingFrameInfo&) = default;
  GPUhdDefault() TrackingFrameInfo& operator=(TrackingFrameInfo&&) = default;

  GPUhd() void print() const;

  float xCoordinate{-999.f};
  float yCoordinate{-999.f};
  float zCoordinate{-999.f};
  float xTrackingFrame{-999.f};
  float alphaTrackingFrame{-999.f};
  std::array<float, 2> positionTrackingFrame = {-999.f, -999.f};
  std::array<float, 3> covarianceTrackingFrame = {-999.f, -999.f, -999.f};

  ClassDefNV(TrackingFrameInfo, 1);
};

} // namespace o2::its

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
