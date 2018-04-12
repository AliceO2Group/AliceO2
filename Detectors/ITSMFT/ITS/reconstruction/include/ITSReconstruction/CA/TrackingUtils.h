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
/// \file CATrackingUtils.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKINGUTILS_H_
#define TRACKINGITSU_INCLUDE_TRACKINGUTILS_H_

#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Definitions.h"

namespace o2
{
namespace ITS
{
namespace CA
{

namespace TrackingUtils
{
GPU_HOST_DEVICE constexpr int4 getEmptyBinsRect() { return int4{ 0, 0, 0, 0 }; }
GPU_DEVICE const int4 getBinsRect(const Cluster&, const int, const float);

float computeCurvature(float x1, float y1, float x2, float y2, float x3, float y3);
float computeCurvatureCentreX(float x1, float y1, float x2, float y2, float x3, float y3);
float computeTanDipAngle(float x1, float y1, float x2, float y2, float z1, float z2);
} // namespace TrackingUtils
} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CATRACKINGUTILS_H_ */
