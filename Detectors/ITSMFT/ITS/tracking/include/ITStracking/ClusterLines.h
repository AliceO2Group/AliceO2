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

#include <array>
#include <vector>
#include "ITStracking/Cluster.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"
#include "GPUCommonMath.h"

namespace o2::its
{
struct Line final {
  GPUhd() Line();
  GPUhd() Line(const Line&);
  Line(std::array<float, 3> firstPoint, std::array<float, 3> secondPoint);
  GPUhd() Line(const float firstPoint[3], const float secondPoint[3]);
  GPUhd() Line(const Tracklet&, const Cluster*, const Cluster*);

  static float getDistanceFromPoint(const Line& line, const std::array<float, 3>& point);
  GPUhd() static float getDistanceFromPoint(const Line& line, const float point[3]);
  static std::array<float, 6> getDCAComponents(const Line& line, const std::array<float, 3> point);
  GPUhd() static void getDCAComponents(const Line& line, const float point[3], float destArray[6]);
  GPUhd() static float getDCA(const Line&, const Line&, const float precision = 1e-14);
  static bool areParallel(const Line&, const Line&, const float precision = 1e-14);
  GPUhd() unsigned char isEmpty() const { return (originPoint[0] == 0.f && originPoint[1] == 0.f && originPoint[2] == 0.f) &&
                                                 (cosinesDirector[0] == 0.f && cosinesDirector[1] == 0.f && cosinesDirector[2] == 0.f); }
  GPUhdi() auto getDeltaROF() const { return rof[1] - rof[0]; }
  GPUhd() void print() const;
  bool operator==(const Line&) const;
  bool operator!=(const Line&) const;
  const short getMinROF() const { return rof[0] < rof[1] ? rof[0] : rof[1]; }

  float originPoint[3], cosinesDirector[3];
  float weightMatrix[6] = {1., 0., 0., 1., 0., 1.};
  // weightMatrix is a symmetric matrix internally stored as
  //    0 --> row = 0, col = 0
  //    1 --> 0,1
  //    2 --> 0,2
  //    3 --> 1,1
  //    4 --> 1,2
  //    5 --> 2,2
  short rof[2];
};

GPUhdi() Line::Line() : weightMatrix{1., 0., 0., 1., 0., 1.}
{
  rof[0] = 0;
  rof[1] = 0;
}

GPUhdi() Line::Line(const Line& other)
{
  for (int i{0}; i < 3; ++i) {
    originPoint[i] = other.originPoint[i];
    cosinesDirector[i] = other.cosinesDirector[i];
  }
  for (int i{0}; i < 6; ++i) {
    weightMatrix[i] = other.weightMatrix[i];
  }
  for (int i{0}; i < 2; ++i) {
    rof[i] = other.rof[i];
  }
}

GPUhdi() Line::Line(const float firstPoint[3], const float secondPoint[3])
{
  for (int i{0}; i < 3; ++i) {
    originPoint[i] = firstPoint[i];
    cosinesDirector[i] = secondPoint[i] - firstPoint[i];
  }

  float inverseNorm{1.f / o2::gpu::CAMath::Sqrt(cosinesDirector[0] * cosinesDirector[0] + cosinesDirector[1] * cosinesDirector[1] +
                                                cosinesDirector[2] * cosinesDirector[2])};

  for (int index{0}; index < 3; ++index) {
    cosinesDirector[index] *= inverseNorm;
  }

  rof[0] = 0;
  rof[1] = 0;
}

GPUhdi() Line::Line(const Tracklet& tracklet, const Cluster* innerClusters, const Cluster* outerClusters)
{
  originPoint[0] = innerClusters[tracklet.firstClusterIndex].xCoordinate;
  originPoint[1] = innerClusters[tracklet.firstClusterIndex].yCoordinate;
  originPoint[2] = innerClusters[tracklet.firstClusterIndex].zCoordinate;

  cosinesDirector[0] = outerClusters[tracklet.secondClusterIndex].xCoordinate - innerClusters[tracklet.firstClusterIndex].xCoordinate;
  cosinesDirector[1] = outerClusters[tracklet.secondClusterIndex].yCoordinate - innerClusters[tracklet.firstClusterIndex].yCoordinate;
  cosinesDirector[2] = outerClusters[tracklet.secondClusterIndex].zCoordinate - innerClusters[tracklet.firstClusterIndex].zCoordinate;

  float inverseNorm{1.f / o2::gpu::CAMath::Sqrt(cosinesDirector[0] * cosinesDirector[0] + cosinesDirector[1] * cosinesDirector[1] +
                                                cosinesDirector[2] * cosinesDirector[2])};

  for (int index{0}; index < 3; ++index) {
    cosinesDirector[index] *= inverseNorm;
  }

  rof[0] = tracklet.rof[0];
  rof[1] = tracklet.rof[1];
}

// static functions:
inline float Line::getDistanceFromPoint(const Line& line, const std::array<float, 3>& point)
{
  float DCASquared{0};
  float cdelta{0};
  for (int i{0}; i < 3; ++i) {
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);
  }
  for (int i{0}; i < 3; ++i) {
    DCASquared += (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta) *
                  (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta);
  }
  return o2::gpu::CAMath::Sqrt(DCASquared);
}

GPUhdi() float Line::getDistanceFromPoint(const Line& line, const float point[3])
{
  float DCASquared{0};
  float cdelta{0};
  for (int i{0}; i < 3; ++i) {
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);
  }
  for (int i{0}; i < 3; ++i) {
    DCASquared += (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta) *
                  (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta);
  }
  return o2::gpu::CAMath::Sqrt(DCASquared);
}

GPUhdi() float Line::getDCA(const Line& firstLine, const Line& secondLine, const float precision)
{
  float normalVector[3];
  normalVector[0] = firstLine.cosinesDirector[1] * secondLine.cosinesDirector[2] -
                    firstLine.cosinesDirector[2] * secondLine.cosinesDirector[1];
  normalVector[1] = -firstLine.cosinesDirector[0] * secondLine.cosinesDirector[2] +
                    firstLine.cosinesDirector[2] * secondLine.cosinesDirector[0];
  normalVector[2] = firstLine.cosinesDirector[0] * secondLine.cosinesDirector[1] -
                    firstLine.cosinesDirector[1] * secondLine.cosinesDirector[0];

  float norm{0.f}, distance{0.f};
  for (int i{0}; i < 3; ++i) {
    norm += normalVector[i] * normalVector[i];
    distance += (secondLine.originPoint[i] - firstLine.originPoint[i]) * normalVector[i];
  }
  if (norm > precision) {
    return o2::gpu::CAMath::Abs(distance / o2::gpu::CAMath::Sqrt(norm));
  } else {
#if defined(__CUDACC__) || defined(__HIPCC__)
    float stdOriginPoint[3];
    for (int i{0}; i < 3; ++i) {
      stdOriginPoint[i] = secondLine.originPoint[1];
    }
#else
    std::array<float, 3> stdOriginPoint = {};
    std::copy_n(secondLine.originPoint, 3, stdOriginPoint.begin());
#endif
    return getDistanceFromPoint(firstLine, stdOriginPoint);
  }
}

GPUhdi() void Line::getDCAComponents(const Line& line, const float point[3], float destArray[6])
{
  float cdelta{0.};
  for (int i{0}; i < 3; ++i) {
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);
  }

  destArray[0] = line.originPoint[0] - point[0] + line.cosinesDirector[0] * cdelta;
  destArray[3] = line.originPoint[1] - point[1] + line.cosinesDirector[1] * cdelta;
  destArray[5] = line.originPoint[2] - point[2] + line.cosinesDirector[2] * cdelta;
  destArray[1] = o2::gpu::CAMath::Sqrt(destArray[0] * destArray[0] + destArray[3] * destArray[3]);
  destArray[2] = o2::gpu::CAMath::Sqrt(destArray[0] * destArray[0] + destArray[5] * destArray[5]);
  destArray[4] = o2::gpu::CAMath::Sqrt(destArray[3] * destArray[3] + destArray[5] * destArray[5]);
}

inline bool Line::operator==(const Line& rhs) const
{
  bool val{false};
  for (int i{0}; i < 3; ++i) {
    val &= this->originPoint[i] == rhs.originPoint[i];
  }
  return val;
}

inline bool Line::operator!=(const Line& rhs) const
{
  bool val;
  for (int i{0}; i < 3; ++i) {
    val &= this->originPoint[i] != rhs.originPoint[i];
  }
  return val;
}

GPUhdi() void Line::print() const
{
  printf("Line: originPoint = (%f, %f, %f), cosinesDirector = (%f, %f, %f), rofs = (%u, %u)\n",
         originPoint[0], originPoint[1], originPoint[2], cosinesDirector[0], cosinesDirector[1], cosinesDirector[2], rof[0], rof[1]);
}

class ClusterLines final
{
 public:
  ClusterLines() = default;
  ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine,
               const bool weight = false);
  ClusterLines(const Line& firstLine, const Line& secondLine);
  void add(const int& lineLabel, const Line& line, const bool& weight = false);
  void computeClusterCentroid();
  void updateROFPoll(const Line&);
  inline std::vector<int>& getLabels()
  {
    return mLabels;
  }
  inline int getSize() const { return mLabels.size(); }
  inline short getROF() const { return mROF; }
  inline std::array<float, 3> getVertex() const { return mVertex; }
  inline std::array<float, 6> getRMS2() const { return mRMS2; }
  inline float getAvgDistance2() const { return mAvgDistance2; }

  bool operator==(const ClusterLines&) const;

 protected:
  std::array<double, 6> mAMatrix;             // AX=B
  std::array<double, 3> mBMatrix;             // AX=B
  std::vector<int> mLabels;                   // labels
  std::array<float, 9> mWeightMatrix = {0.f}; // weight matrix
  std::array<float, 3> mVertex = {0.f};       // cluster centroid position
  std::array<float, 6> mRMS2 = {0.f};         // symmetric matrix: diagonal is RMS2
  float mAvgDistance2 = 0.f;                  // substitute for chi2
  int mROFWeight = 0;                         // rof weight for voting
  short mROF = -1;                            // rof
};

} // namespace o2::its
#endif /* O2_ITS_CLUSTERLINES_H */
