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
/// \file Cluster.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CACLUSTER_H_
#define TRACKINGITSU_INCLUDE_CACLUSTER_H_

#ifndef __OPENCL__
#include <array>
#endif

#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/IndexTableUtils.h"

namespace o2
{
namespace its
{

struct Cluster final {
  Cluster(const float x, const float y, const float z, const int idx);
  Cluster(const int, const Cluster&);
  Cluster(const int, const float3&, const Cluster&);

  float xCoordinate;
  float yCoordinate;
  float zCoordinate;
  float phiCoordinate;
  float rCoordinate;
  int clusterId;
  int indexTableBinIndex;
};

struct TrackingFrameInfo {
  TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, GPUArray<float, 2>&& posTF, GPUArray<float, 3>&& covTF);
  TrackingFrameInfo() = default;

  float xCoordinate;
  float yCoordinate;
  float zCoordinate;
  float xTrackingFrame;
  float alphaTrackingFrame;
  GPUArray<float, 2> positionTrackingFrame = { -1., -1. };
  GPUArray<float, 3> covarianceTrackingFrame = { 999., 999., 999. };
};
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CACLUSTER_H_ */
