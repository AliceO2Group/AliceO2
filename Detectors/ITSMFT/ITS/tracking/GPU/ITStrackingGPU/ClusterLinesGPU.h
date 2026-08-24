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

/// \file ClusterLinesGPU.h
/// \brief device-side line + N-line vertex fit for the GPU seeding vertexer.

#ifndef O2_ITS_CLUSTERLINES_GPU_H
#define O2_ITS_CLUSTERLINES_GPU_H

#include "DataFormatsITS/TimeEstBC.h"
#include "GPUCommonDef.h"
#include "GPUCommonMath.h"
#include "ITStracking/LineProjection.h"

namespace o2::its::gpu
{

using LineWindow = o2::its::LineWindow;

struct LineProjSoA {
  float* z{nullptr};              // projected z at the beamline; sort key and binary-search key, kept dense
  o2::its::TimeEstBC* t{nullptr}; // time interval of the line
  int* idx{nullptr};              // sorted slot -> original line index
  int* rof{nullptr};              // ROF of the line
};

struct VertexCand {
  float x, y, z;
  float rms2[6];
  float avgDist2;
  int nGood;
  float seed[3];
  o2::its::TimeEstBC time;
  int size;
  uint8_t ok;   // 1 if the candidate passed the fit cuts
  uint8_t keep; // 1 if it survived duplicate suppression (subset of ok)
  uint8_t fine;
};

// Device-side line: origin point + unit direction, with a time stamp
struct GPULine {
  GPUhdDefault() GPULine() = default;

  GPUhdi() GPULine(const float origin[3], const float direction[3], const o2::its::TimeEstBC& t) : mTime(t)
  {
    const float norm = o2::gpu::GPUCommonMath::Sqrt(direction[0] * direction[0] +
                                                    direction[1] * direction[1] +
                                                    direction[2] * direction[2]);
    const float inv = norm > 0.f ? 1.f / norm : 0.f;
    for (int i = 0; i < 3; ++i) {
      originPoint[i] = origin[i];
      cosinesDirector[i] = direction[i] * inv;
    }
  }

  // Squared distance from a point to the (infinite) line: |delta - (delta.u) u|^2
  GPUhdi() static float getDistance2FromPoint(const GPULine& line, const float point[3])
  {
    float delta[3];
    float proj = 0.f;
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

  GPUhdi() static void getDCAComponents(const GPULine& line, const float point[3], float out[6])
  {
    float delta[3];
    float proj = 0.f;
    for (int i = 0; i < 3; ++i) {
      delta[i] = line.originPoint[i] - point[i];
      proj += delta[i] * line.cosinesDirector[i];
    }
    float r[3];
    for (int i = 0; i < 3; ++i) {
      r[i] = delta[i] - proj * line.cosinesDirector[i];
    }
    out[0] = r[0];                                      // (0,0) XX
    out[1] = o2::gpu::GPUCommonMath::Hypot(r[0], r[1]); // (0,1) XY
    out[2] = r[1];                                      // (1,1) YY
    out[3] = o2::gpu::GPUCommonMath::Hypot(r[0], r[2]); // (0,2) XZ
    out[4] = o2::gpu::GPUCommonMath::Hypot(r[1], r[2]); // (1,2) YZ
    out[5] = r[2];                                      // (2,2) ZZ
  }

  float originPoint[3] = {0.f, 0.f, 0.f};
  float cosinesDirector[3] = {0.f, 0.f, 0.f};
  o2::its::TimeEstBC mTime;
};

class GPUClusterLinesFit
{
 public:
  GPUhdDefault() GPUClusterLinesFit() = default;

  // Add one line's contribution: A_ij += (delta_ij*|d|^2 - d_i*d_j)/|d|^2,
  // b_i += (d_i*(d.o) - |d|^2*o_i)/|d|^2. For a unit director |d|^2 == 1.
  GPUhdi() void add(const GPULine& line)
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

  // Solve the symmetric system and write the vertex (= -A^-1 B)
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

  GPUhdi() void addResidual(const GPULine& line, const float vertex[3])
  {
    float dca[6];
    GPULine::getDCAComponents(line, vertex, dca);
    const float d2 = GPULine::getDistance2FromPoint(line, vertex);
    ++mResidualCount;
    const float inv = 1.f / static_cast<float>(mResidualCount);
    for (int i = 0; i < 6; ++i) {
      mRMS2[i] += (dca[i] - mRMS2[i]) * inv;
    }
    mAvgDistance2 += (d2 - mAvgDistance2) * inv;
  }

  GPUhdi() int getNContributors() const { return mNContributors; }
  GPUhdi() const float* getRMS2() const { return mRMS2; } // Packed symmetric covariance in {XX, XY, YY, XZ, YZ, ZZ} order
  GPUhdi() float getAvgDistance2() const { return mAvgDistance2; }
  GPUhdi() const o2::its::TimeEstBC& getTimeStamp() const { return mTime; }

 private:
  double mA[6] = {0., 0., 0., 0., 0., 0.};
  double mB[3] = {0., 0., 0.};
  int mNContributors = 0;
  float mRMS2[6] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float mAvgDistance2 = 0.f;
  int mResidualCount = 0;
  o2::its::TimeEstBC mTime;
};

} // namespace o2::its::gpu

#endif /* O2_ITS_CLUSTERLINES_GPU_H */
