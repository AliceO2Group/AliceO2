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

Line::Line(const Tracklet& tracklet, const Cluster* innerClusters, const Cluster* outerClusters) : mTime(tracklet.mTime)
{
  const auto& inner = innerClusters[tracklet.firstClusterIndex];
  const auto& outer = outerClusters[tracklet.secondClusterIndex];

  originPoint = SVector3f(inner.xCoordinate, inner.yCoordinate, inner.zCoordinate);
  cosinesDirector = SVector3f(outer.xCoordinate - inner.xCoordinate,
                              outer.yCoordinate - inner.yCoordinate,
                              outer.zCoordinate - inner.zCoordinate);
  cosinesDirector /= std::sqrt(ROOT::Math::Dot(cosinesDirector, cosinesDirector));
}

float Line::getDistance2FromPoint(const Line& line, const std::array<float, 3>& point)
{
  const SVector3f p(point.data(), 3);
  const SVector3f delta = p - line.originPoint;
  const float proj = ROOT::Math::Dot(delta, line.cosinesDirector);
  const SVector3f residual = delta - proj * line.cosinesDirector;
  return ROOT::Math::Dot(residual, residual);
}

float Line::getDistanceFromPoint(const Line& line, const std::array<float, 3>& point)
{
  return std::sqrt(getDistance2FromPoint(line, point));
}

float Line::getDCA2(const Line& firstLine, const Line& secondLine, const float precision)
{
  const SVector3f n = ROOT::Math::Cross(firstLine.cosinesDirector, secondLine.cosinesDirector);
  const float norm2 = ROOT::Math::Dot(n, n);

  if (norm2 <= precision * precision) {
    // lines are parallel, fall back to point-to-line distance
    const SVector3f d = secondLine.originPoint - firstLine.originPoint;
    const float proj = ROOT::Math::Dot(d, firstLine.cosinesDirector);
    const SVector3f residual = d - proj * firstLine.cosinesDirector;
    return ROOT::Math::Dot(residual, residual);
  }

  const SVector3f delta = secondLine.originPoint - firstLine.originPoint;
  const float numerator = ROOT::Math::Dot(delta, n);
  return (numerator * numerator) / norm2;
}

float Line::getDCA(const Line& firstLine, const Line& secondLine, const float precision)
{
  return std::sqrt(getDCA2(firstLine, secondLine, precision));
}

Line::SMatrix3f Line::getDCAComponents(const Line& line, const std::array<float, 3>& point)
{
  const SVector3f p(point.data(), 3);
  const SVector3f delta = line.originPoint - p;
  const float proj = ROOT::Math::Dot(line.cosinesDirector, delta);
  const SVector3f residual = delta - proj * line.cosinesDirector;

  // symmetric 3x3: diagonal = residual components, off-diagonal = 2D projected distances
  SMatrix3f m;
  m(0, 0) = residual(0);
  m(1, 1) = residual(1);
  m(2, 2) = residual(2);
  m(0, 1) = std::hypot(m(0, 0), m(1, 1));
  m(0, 2) = std::hypot(m(0, 0), m(2, 2));
  m(1, 2) = std::hypot(m(1, 1), m(2, 2));
  return m;
}

bool Line::isEmpty() const noexcept
{
  return ROOT::Math::Dot(originPoint, originPoint) == 0.f &&
         ROOT::Math::Dot(cosinesDirector, cosinesDirector) == 0.f;
}

void Line::print() const
{
  LOGP(info, "\tLine: originPoint = ({}, {}, {}), cosinesDirector = ({}, {}, {}) ts={}+-{}",
       originPoint(0), originPoint(1), originPoint(2),
       cosinesDirector(0), cosinesDirector(1), cosinesDirector(2),
       mTime.getTimeStamp(), mTime.getTimeStampError());
}

// Accumulate the weighted normal equation contributions (A matrix and B vector)
// from a single line into the running sums. The covariance is assumed to be
// diagonal and uniform ({1,1,1}) so the weights simplify accordingly.
// The A matrix entry (i,j) = (delta_ij - d_i*d_j) / det, and the B vector
// entry b_i = sum_j d_j*(d_j*o_i - d_i*o_j) / det, where d = cosinesDirector
// and o = originPoint.
void ClusterLines::accumulate(const Line& line)
{
  const ROOT::Math::SVector<double, 3> d(line.cosinesDirector(0), line.cosinesDirector(1), line.cosinesDirector(2));
  const ROOT::Math::SVector<double, 3> o(line.originPoint(0), line.originPoint(1), line.originPoint(2));

  // == 1 for normalised directors, kept for generality
  const double det = ROOT::Math::Dot(d, d);

  // A matrix (symmetric): A_ij = (delta_ij * |d|^2 - d_i * d_j) / det
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      mAMatrix(i, j) += ((i == j ? det : 0.) - d(i) * d(j)) / det;
    }
  }

  // B vector: b_i = (d_i * dot(d,o) - |d|^2 * o_i) / det
  const double dDotO = ROOT::Math::Dot(d, o);
  for (int i = 0; i < 3; ++i) {
    mBMatrix(i) += (d(i) * dDotO - det * o(i)) / det;
  }
}

ClusterLines::ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine) : mTime(firstLine.mTime)
{
  mTime += secondLine.mTime;

  mLabels.push_back(firstLabel);
  if (secondLabel > 0) {
    mLabels.push_back(secondLabel); // don't add info in case of beamline used
  }

  accumulate(firstLine);
  accumulate(secondLine);
  computeClusterCentroid();

  // RMS2: running mean update
  mRMS2 = Line::getDCAComponents(firstLine, mVertex);
  const auto tmpRMS2 = Line::getDCAComponents(secondLine, mVertex);
  mRMS2 += (tmpRMS2 - mRMS2) * (1.f / static_cast<float>(getSize()));

  // AvgDistance2
  mAvgDistance2 = Line::getDistance2FromPoint(firstLine, mVertex);
  mAvgDistance2 += (Line::getDistance2FromPoint(secondLine, mVertex) - mAvgDistance2) / (float)getSize();
}

ClusterLines::ClusterLines(gsl::span<const int> lineLabels, gsl::span<const Line> lines)
{
  if (lineLabels.size() < 2) {
    return;
  }

  mLabels.reserve(lineLabels.size());
  mTime = lines[lineLabels[0]].mTime;
  for (size_t index = 0; index < lineLabels.size(); ++index) {
    const auto lineLabel = lineLabels[index];
    if (index > 0) {
      mTime += lines[lineLabel].mTime;
    }
    mLabels.push_back(lineLabel);
    accumulate(lines[lineLabel]);
  }

  computeClusterCentroid();
  if (!mIsValid) {
    return;
  }

  mRMS2 = Line::getDCAComponents(lines[lineLabels[0]], mVertex);
  mAvgDistance2 = Line::getDistance2FromPoint(lines[lineLabels[0]], mVertex);
  for (size_t index = 1; index < lineLabels.size(); ++index) {
    const auto lineLabel = lineLabels[index];
    const auto tmpRMS2 = Line::getDCAComponents(lines[lineLabel], mVertex);
    mRMS2 += (tmpRMS2 - mRMS2) * (1.f / static_cast<float>(index + 1));
    mAvgDistance2 += (Line::getDistance2FromPoint(lines[lineLabel], mVertex) - mAvgDistance2) / static_cast<float>(index + 1);
  }
}

void ClusterLines::add(const int lineLabel, const Line& line)
{
  mTime += line.mTime;
  mLabels.push_back(lineLabel);

  accumulate(line);
  computeClusterCentroid();
  mAvgDistance2 += (Line::getDistance2FromPoint(line, mVertex) - mAvgDistance2) / (float)getSize();
}

void ClusterLines::computeClusterCentroid()
{
  // Solve the 3x3 symmetric linear system AX = -B using SMatrix inversion.
  // Invert() returns false if the matrix is singular or ill-conditioned.
  SMatrix3 invA{mAMatrix};
  mIsValid = invA.Invert();
  if (!mIsValid) {
    return;
  }

  SVector3 result = invA * mBMatrix;
  mVertex[0] = static_cast<float>(-result(0));
  mVertex[1] = static_cast<float>(-result(1));
  mVertex[2] = static_cast<float>(-result(2));
}

bool ClusterLines::operator==(const ClusterLines& rhs) const noexcept
{
  return mRMS2 == rhs.mRMS2 &&
         mVertex == rhs.mVertex &&
         mLabels == rhs.mLabels &&
         mAvgDistance2 == rhs.mAvgDistance2;
}

} // namespace o2::its
