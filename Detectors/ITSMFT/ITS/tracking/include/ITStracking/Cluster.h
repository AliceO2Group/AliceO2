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

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#endif

#include "GPUCommonRtypes.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"

namespace o2
{
namespace its
{

class IndexTableUtils;

struct Cluster final {
  Cluster() = default;
  Cluster(const float x, const float y, const float z, const int idx);
  Cluster(const int, const IndexTableUtils& utils, const Cluster&);
  Cluster(const int, const float3&, const IndexTableUtils& utils, const Cluster&);
  void Init(const int, const float3&, const IndexTableUtils& utils, const Cluster&);
  bool operator==(const Cluster&) const;
  GPUhd() void print() const;

  float xCoordinate;      // = -999.f;
  float yCoordinate;      // = -999.f;
  float zCoordinate;      // = -999.f;
  float phi;              // = -999.f;
  float radius;           // = -999.f;
  int clusterId;          // = -1;
  int indexTableBinIndex; // = -1;

  ClassDefNV(Cluster, 1);
};

GPUhdi() void Cluster::print() const
{
  printf("Cluster: %f %f %f %f %f %d %d\n", xCoordinate, yCoordinate, zCoordinate, phi, radius, clusterId, indexTableBinIndex);
}

struct TrackingFrameInfo {
  TrackingFrameInfo() = default;
  TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, GPUArray<float, 2>&& posTF, GPUArray<float, 3>&& covTF);

  float xCoordinate;
  float yCoordinate;
  float zCoordinate;
  float xTrackingFrame;
  float alphaTrackingFrame;
  GPUArray<float, 2> positionTrackingFrame = {-1., -1.};
  GPUArray<float, 3> covarianceTrackingFrame = {999., 999., 999.};

  ClassDefNV(TrackingFrameInfo, 1);
};
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
