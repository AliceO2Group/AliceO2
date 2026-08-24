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

#ifndef O2_ITS_CLUSTERLINES_H
#define O2_ITS_CLUSTERLINES_H

#include <gsl/span>
#include <vector>
#include <array>
#include "ITStracking/Cluster.h"
#include "ITSMFTTracking/Constants.h"
#include "ITStracking/Tracklet.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"

namespace o2::its
{

struct Line final {
  GPUhdDefault() Line() = default;
  Line(const Tracklet&, const Cluster*, const Cluster*);
  GPUhdi() Line(const float origin[3], const float direction[3], const TimeEstBC& time) : mTime(time)
  {
    float norm2 = 0.f;
    for (int i = 0; i < 3; ++i) {
      norm2 += direction[i] * direction[i];
    }
    const float inv = norm2 > 0.f ? 1.f / o2::gpu::GPUCommonMath::Sqrt(norm2) : 0.f;
    for (int i = 0; i < 3; ++i) {
      originPoint[i] = origin[i];
      cosinesDirector[i] = direction[i] * inv;
    }
  }
  Line(const std::array<float, 3>& origin, const std::array<float, 3>& direction, const TimeEstBC& time)
    : Line(origin.data(), direction.data(), time) {}
  bool operator==(const Line&) const = default;

  GPUhdi() static float getDistance2FromPoint(const Line& line, const float point[3])
  {
    float delta[3], proj = 0.f;
    for (int i = 0; i < 3; ++i) {
      delta[i] = point[i] - line.originPoint[i];
      proj += delta[i] * line.cosinesDirector[i];
    }
    float d2 = 0.f;
    for (int i = 0; i < 3; ++i) {
      const float residual = delta[i] - proj * line.cosinesDirector[i];
      d2 += residual * residual;
    }
    return d2;
  }

  /// Packed symmetric DCA components in {XX, XY, YY, XZ, YZ, ZZ} order
  GPUhdi() static void getDCAComponents(const Line& line, const float point[3], float out[6])
  {
    float delta[3], proj = 0.f;
    for (int i = 0; i < 3; ++i) {
      delta[i] = line.originPoint[i] - point[i];
      proj += delta[i] * line.cosinesDirector[i];
    }
    float r[3];
    for (int i = 0; i < 3; ++i) {
      r[i] = delta[i] - proj * line.cosinesDirector[i];
    }
    out[0] = r[0];
    out[1] = o2::gpu::GPUCommonMath::Hypot(r[0], r[1]);
    out[2] = r[1];
    out[3] = o2::gpu::GPUCommonMath::Hypot(r[0], r[2]);
    out[4] = o2::gpu::GPUCommonMath::Hypot(r[1], r[2]);
    out[5] = r[2];
  }

  static float getDistance2FromPoint(const Line& line, const std::array<float, 3>& point)
  {
    return getDistance2FromPoint(line, point.data());
  }
  static float getDistanceFromPoint(const Line& line, const std::array<float, 3>& point);
  static float getDCA2(const Line&, const Line&, const float precision = constants::Tolerance);
  static float getDCA(const Line&, const Line&, const float precision = constants::Tolerance);
  GPUhdi() bool isEmpty() const noexcept
  {
    return originPoint[0] == 0.f && originPoint[1] == 0.f && originPoint[2] == 0.f &&
           cosinesDirector[0] == 0.f && cosinesDirector[1] == 0.f && cosinesDirector[2] == 0.f;
  }
  void print() const;

  float originPoint[3] = {0.f, 0.f, 0.f};
  float cosinesDirector[3] = {0.f, 0.f, 0.f};
  TimeEstBC mTime;
};

/// Least-squares vertex fit over a set of lines (the normal equations AX = -B).
class ClusterLines final
{
 public:
  GPUhdDefault() ClusterLines() = default;
  /// Fit over lines[lineIndices[i]]; lineIndices must be sorted and unique.
  ClusterLines(gsl::span<const int> lineIndices, gsl::span<const Line> lines);

  /// Accumulate one line into the normal equations and the time estimate.
  GPUhdi() void add(const Line& line)
  {
    const double d0 = line.cosinesDirector[0], d1 = line.cosinesDirector[1], d2 = line.cosinesDirector[2];
    const double o0 = line.originPoint[0], o1 = line.originPoint[1], o2 = line.originPoint[2];
    const double det = d0 * d0 + d1 * d1 + d2 * d2; // == 1 for a normalised director
    if (det <= 0.) {
      return;
    }
    if (mNContributors <= 0) {
      mTime = line.mTime;
    } else {
      mTime += line.mTime;
    }
    mA[0] += (det - d0 * d0) / det;
    mA[1] += (-d0 * d1) / det;
    mA[2] += (-d0 * d2) / det;
    mA[3] += (det - d1 * d1) / det;
    mA[4] += (-d1 * d2) / det;
    mA[5] += (det - d2 * d2) / det;
    const double dDotO = d0 * o0 + d1 * o1 + d2 * o2;
    mB[0] += (d0 * dDotO - det * o0) / det;
    mB[1] += (d1 * dDotO - det * o1) / det;
    mB[2] += (d2 * dDotO - det * o2) / det;
    ++mNContributors;
  }

  /// Solve the symmetric 3x3 system into an external vertex (= -A^-1 B)
  GPUhdi() bool solve(float vertex[3]) const
  {
    const double a = mA[0], b = mA[1], c = mA[2], d = mA[3], e = mA[4], f = mA[5];
    const double c00 = d * f - e * e;
    const double c01 = c * e - b * f;
    const double c02 = b * e - c * d;
    const double c11 = a * f - c * c;
    const double c12 = b * c - a * e;
    const double c22 = a * d - b * b;
    const double det = a * c00 + b * c01 + c * c02;
    if (o2::gpu::GPUCommonMath::Abs(det) < 1.e-12) {
      return false;
    }
    const double invDet = 1. / det;
    const double x0 = (c00 * mB[0] + c01 * mB[1] + c02 * mB[2]) * invDet;
    const double x1 = (c01 * mB[0] + c11 * mB[1] + c12 * mB[2]) * invDet;
    const double x2 = (c02 * mB[0] + c12 * mB[1] + c22 * mB[2]) * invDet;
    vertex[0] = static_cast<float>(-x0);
    vertex[1] = static_cast<float>(-x1);
    vertex[2] = static_cast<float>(-x2);
    return true;
  }

  /// Solve into the stored vertex, setting the validity flag
  GPUhdi() void computeClusterCentroid() { mIsValid = solve(mVertex); }

  /// Running-mean update of the RMS2 and average distance about the stored vertex
  GPUhdi() void addResidual(const Line& line) { addResidual(line, mVertex); }

  /// Running-mean update about an externally solved vertex
  GPUhdi() void addResidual(const Line& line, const float vertex[3])
  {
    float dca[6];
    Line::getDCAComponents(line, vertex, dca);
    const float d2 = Line::getDistance2FromPoint(line, vertex);
    ++mResidualCount;
    const float inv = 1.f / static_cast<float>(mResidualCount);
    for (int i = 0; i < 6; ++i) {
      mRMS2[i] += (dca[i] - mRMS2[i]) * inv;
    }
    mAvgDistance2 += (d2 - mAvgDistance2) * inv;
  }

  GPUhdi() bool isValid() const noexcept { return mIsValid; }
  GPUhdi() const float* getVertex() const noexcept { return mVertex; }
  GPUhdi() const float* getRMS2() const noexcept { return mRMS2; } // {XX, XY, YY, XZ, YZ, ZZ}
  GPUhdi() float getAvgDistance2() const noexcept { return mAvgDistance2; }
  GPUhdi() int getSize() const noexcept { return mNContributors; }
  GPUhdi() int getNContributors() const noexcept { return mNContributors; }
  GPUhdi() const TimeEstBC& getTimeStamp() const noexcept { return mTime; }
  GPUhdi() float getR2() const noexcept { return (mVertex[0] * mVertex[0]) + (mVertex[1] * mVertex[1]); }
  GPUhdi() float getR() const noexcept { return o2::gpu::GPUCommonMath::Sqrt(getR2()); }
  bool operator==(const ClusterLines& rhs) const noexcept;

 private:
  double mA[6] = {0., 0., 0., 0., 0., 0.}; // AX=B, packed symmetric normal matrix
  double mB[3] = {0., 0., 0.};             // AX=B, right-hand side
  float mVertex[3] = {0.f, 0.f, 0.f};      // cluster centroid position
  float mRMS2[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float mAvgDistance2 = 0.f;
  int mNContributors = 0;
  int mResidualCount = 0;
  bool mIsValid = false; // true if the linear system was solved successfully
  TimeEstBC mTime;
};

} // namespace o2::its
#endif /* O2_ITS_CLUSTERLINES_H */
