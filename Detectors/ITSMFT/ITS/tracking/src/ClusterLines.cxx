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

#include <cmath>
#include "Framework/Logger.h"
#include "ITStracking/ClusterLines.h"

namespace o2::its
{

Line::Line(const Tracklet& tracklet, const Cluster* innerClusters, const Cluster* outerClusters)
{
  const auto& inner = innerClusters[tracklet.firstClusterIndex];
  const auto& outer = outerClusters[tracklet.secondClusterIndex];

  const float origin[3] = {inner.xCoordinate, inner.yCoordinate, inner.zCoordinate};
  const float direction[3] = {outer.xCoordinate - inner.xCoordinate,
                              outer.yCoordinate - inner.yCoordinate,
                              outer.zCoordinate - inner.zCoordinate};
  *this = Line(origin, direction, tracklet.mTime);
}

float Line::getDistanceFromPoint(const Line& line, const std::array<float, 3>& point)
{
  return std::sqrt(getDistance2FromPoint(line, point.data()));
}

float Line::getDCA2(const Line& firstLine, const Line& secondLine, const float precision)
{
  const auto& a = firstLine.cosinesDirector;
  const auto& b = secondLine.cosinesDirector;
  const float n[3] = {a[1] * b[2] - a[2] * b[1],
                      a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0]};
  const float norm2 = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

  float d[3];
  for (int i = 0; i < 3; ++i) {
    d[i] = secondLine.originPoint[i] - firstLine.originPoint[i];
  }

  if (norm2 <= precision * precision) {
    // lines are parallel, fall back to point-to-line distance
    float proj = 0.f;
    for (int i = 0; i < 3; ++i) {
      proj += d[i] * a[i];
    }
    float res2 = 0.f;
    for (int i = 0; i < 3; ++i) {
      const float residual = d[i] - proj * a[i];
      res2 += residual * residual;
    }
    return res2;
  }

  const float numerator = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
  return (numerator * numerator) / norm2;
}

float Line::getDCA(const Line& firstLine, const Line& secondLine, const float precision)
{
  return std::sqrt(getDCA2(firstLine, secondLine, precision));
}

void Line::print() const
{
  LOGP(info, "\tLine: originPoint = ({}, {}, {}), cosinesDirector = ({}, {}, {}) ts={}+-{}",
       originPoint[0], originPoint[1], originPoint[2],
       cosinesDirector[0], cosinesDirector[1], cosinesDirector[2],
       mTime.getTimeStamp(), mTime.getTimeStampError());
}

ClusterLines::ClusterLines(gsl::span<const int> lineIndices, gsl::span<const Line> lines)
{
  if (lineIndices.size() < 2) {
    return;
  }
  for (const auto idx : lineIndices) {
    add(lines[idx]);
  }
  computeClusterCentroid();
  if (!mIsValid) {
    return;
  }
  for (const auto idx : lineIndices) {
    addResidual(lines[idx]);
  }
}

bool ClusterLines::operator==(const ClusterLines& rhs) const noexcept
{
  for (int i = 0; i < 6; ++i) {
    if (mRMS2[i] != rhs.mRMS2[i]) {
      return false;
    }
  }
  for (int i = 0; i < 3; ++i) {
    if (mVertex[i] != rhs.mVertex[i]) {
      return false;
    }
  }
  return mNContributors == rhs.mNContributors && mAvgDistance2 == rhs.mAvgDistance2;
}

} // namespace o2::its
