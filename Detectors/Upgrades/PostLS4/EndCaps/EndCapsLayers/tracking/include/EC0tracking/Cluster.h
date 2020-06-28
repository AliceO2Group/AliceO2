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

#ifndef TRACKINGEC0__INCLUDE_CACLUSTER_H_
#define TRACKINGEC0__INCLUDE_CACLUSTER_H_

#ifndef __OPENCL__
#include <array>
#endif

#include "EC0tracking/Definitions.h"
#include "EC0tracking/MathUtils.h"
#include "EC0tracking/IndexTableUtils.h"

namespace o2
{
namespace ecl
{

struct Cluster final {
  Cluster() = default;
  Cluster(const float x, const float y, const float z, const int idx);
  Cluster(const int, const Cluster&);
  Cluster(const int, const float3&, const Cluster&);
  void Init(const int, const float3&, const Cluster&);

  float xCoordinate;      // = -999.f;
  float yCoordinate;      // = -999.f;
  float zCoordinate;      // = -999.f;
  float phiCoordinate;    // = -999.f;
  float rCoordinate;      // = -999.f;
  int clusterId;          // = -1;
  int indexTableBinIndex; // = -1;
};

struct TrackingFrameInfo {
  TrackingFrameInfo(float x, float y, float z, float xTF, float alpha, GPUArrayEC0<float, 2>&& posTF, GPUArrayEC0<float, 3>&& covTF);
  TrackingFrameInfo() = default;

  float xCoordinate;
  float yCoordinate;
  float zCoordinate;
  float xTrackingFrame;
  float alphaTrackingFrame;
  GPUArrayEC0<float, 2> positionTrackingFrame = {-1., -1.};
  GPUArrayEC0<float, 3> covarianceTrackingFrame = {999., 999., 999.};
};
} // namespace ecl
} // namespace o2

#endif /* TRACKINGEC0__INCLUDE_CACLUSTER_H_ */
