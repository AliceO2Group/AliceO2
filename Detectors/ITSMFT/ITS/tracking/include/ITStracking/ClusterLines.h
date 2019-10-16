// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITSMFT_TRACKING_LINE_H_
#define O2_ITSMFT_TRACKING_LINE_H_

#include <array>
#include <vector>
#include "ITStracking/Cluster.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Tracklet.h"
#include "GPUCommonMath.h"

namespace o2
{
namespace its
{

struct Line final {
  GPUhd() Line();
  GPUhd() Line(const Line&);
  Line(std::array<float, 3> firstPoint, std::array<float, 3> secondPoint);
  GPUhd() Line(const float firstPoint[3], const float secondPoint[3]);
  GPUhd() Line(const Tracklet&, const Cluster*, const Cluster*);

  inline static float getDistanceFromPoint(const Line& line, const std::array<float, 3>& point);
  GPUhd() static float getDistanceFromPoint(const Line& line, const float point[3]);
  static std::array<float, 6> getDCAComponents(const Line& line, const std::array<float, 3> point);
  GPUhd() static void getDCAComponents(const Line& line, const float point[3], float destArray[6]);
  GPUhd() static float getDCA(const Line&, const Line&, const float precision = 1e-14);
  static bool areParallel(const Line&, const Line&, const float precision = 1e-14);

  float originPoint[3], cosinesDirector[3];         // std::array<float, 3> originPoint, cosinesDirector;
  float weightMatrix[6] = {1., 0., 0., 1., 0., 1.}; // std::array<float, 6> weightMatrix;
  unsigned char isEmpty = false;
  // weightMatrix is a symmetric matrix internally stored as
  //    0 --> row = 0, col = 0
  //    1 --> 0,1
  //    2 --> 0,2
  //    3 --> 1,1
  //    4 --> 1,2
  //    5 --> 2,2
};

inline Line::Line() : weightMatrix{1., 0., 0., 1., 0., 1.}
{
  isEmpty = true;
}

inline Line::Line(const Line& other)
{
  isEmpty = other.isEmpty;
  for (int i{0}; i < 3; ++i) {
    originPoint[i] = other.originPoint[i];
    cosinesDirector[i] = other.cosinesDirector[i];
  }
  for (int i{0}; i < 6; ++i) {
    weightMatrix[i] = other.weightMatrix[i];
  }
}

inline Line::Line(const float firstPoint[3], const float secondPoint[3])
{
  for (int i{0}; i < 3; ++i) {
    originPoint[i] = firstPoint[i];
    cosinesDirector[i] = secondPoint[i] - firstPoint[i];
  }

  float inverseNorm{1.f / gpu::CAMath::Sqrt(cosinesDirector[0] * cosinesDirector[0] + cosinesDirector[1] * cosinesDirector[1] +
                                            cosinesDirector[2] * cosinesDirector[2])};

  for (int index{0}; index < 3; ++index)
    cosinesDirector[index] *= inverseNorm;
}

inline Line::Line(const Tracklet& tracklet, const Cluster* innerClusters, const Cluster* outerClusters)
{
  originPoint[0] = innerClusters[tracklet.firstClusterIndex].xCoordinate;
  originPoint[1] = innerClusters[tracklet.firstClusterIndex].yCoordinate;
  originPoint[2] = innerClusters[tracklet.firstClusterIndex].zCoordinate;

  cosinesDirector[0] = outerClusters[tracklet.secondClusterIndex].xCoordinate - innerClusters[tracklet.firstClusterIndex].xCoordinate;
  cosinesDirector[1] = outerClusters[tracklet.secondClusterIndex].yCoordinate - innerClusters[tracklet.firstClusterIndex].yCoordinate;
  cosinesDirector[2] = outerClusters[tracklet.secondClusterIndex].zCoordinate - innerClusters[tracklet.firstClusterIndex].zCoordinate;

  float inverseNorm{1.f / gpu::CAMath::Sqrt(cosinesDirector[0] * cosinesDirector[0] + cosinesDirector[1] * cosinesDirector[1] +
                                            cosinesDirector[2] * cosinesDirector[2])};

  for (int index{0}; index < 3; ++index)
    cosinesDirector[index] *= inverseNorm;
}

// static functions
inline float Line::getDistanceFromPoint(const Line& line, const std::array<float, 3>& point)
{
  float DCASquared{0};
  float cdelta{0};
  for (int i{0}; i < 3; ++i)
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);
  for (int i{0}; i < 3; ++i) {
    DCASquared += (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta) *
                  (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta);
  }
  return gpu::CAMath::Sqrt(DCASquared);
}

inline float Line::getDistanceFromPoint(const Line& line, const float point[3])
{
  float DCASquared{0};
  float cdelta{0};
  for (int i{0}; i < 3; ++i)
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);
  for (int i{0}; i < 3; ++i) {
    DCASquared += (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta) *
                  (line.originPoint[i] - point[i] + line.cosinesDirector[i] * cdelta);
  }
  return gpu::CAMath::Sqrt(DCASquared);
}

inline float Line::getDCA(const Line& firstLine, const Line& secondLine, const float precision)
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
    return gpu::CAMath::Abs(distance / gpu::CAMath::Sqrt(norm));
  } else {
#if defined(__CUDACC__) || defined(__HIPCC__)
    float stdOriginPoint[3];
    for (int i{0}; i < 3; ++i) {
      stdOriginPoint[i] = secondLine.originPoint[1];
    }
#else
    std::array<float, 3> stdOriginPoint;
    std::copy_n(secondLine.originPoint, 3, stdOriginPoint.begin());
#endif
    return getDistanceFromPoint(firstLine, stdOriginPoint);
  }
}

inline void Line::getDCAComponents(const Line& line, const float point[3], float destArray[6])
{
  float cdelta{0.};
  for (int i{0}; i < 3; ++i)
    cdelta -= line.cosinesDirector[i] * (line.originPoint[i] - point[i]);

  destArray[0] = line.originPoint[0] - point[0] + line.cosinesDirector[0] * cdelta;
  destArray[3] = line.originPoint[1] - point[1] + line.cosinesDirector[1] * cdelta;
  destArray[5] = line.originPoint[2] - point[2] + line.cosinesDirector[2] * cdelta;
  destArray[1] = std::sqrt(destArray[0] * destArray[0] + destArray[3] * destArray[3]);
  destArray[2] = std::sqrt(destArray[0] * destArray[0] + destArray[5] * destArray[5]);
  destArray[4] = std::sqrt(destArray[3] * destArray[3] + destArray[5] * destArray[5]);
}

///

class ClusterLines final
{
 public:
  ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine,
               const bool weight = false);
  ClusterLines(const Line& firstLine, const Line& secondLine);
  void add(const int& lineLabel, const Line& line, const bool& weight = false);
  void computeClusterCentroid();
  inline std::vector<int>& getLabels() { return mLabels; }
  inline int getSize() const { return mLabels.size(); }
  inline std::array<float, 3> getVertex() const { return mVertex; }
  inline std::array<float, 6> getRMS2() const { return mRMS2; }
  inline float getAvgDistance2() const { return mAvgDistance2; }

#ifdef _ALLOW_DEBUG_TREES_ITS_
  inline std::vector<Line> getLines()
  {
    return mLines;
  }
#endif

 protected:
  std::array<float, 6> mAMatrix;         // AX=B
  std::array<float, 3> mBMatrix;         // AX=B
  std::vector<int> mLabels;              // labels
  std::array<float, 3> mVertexCandidate; // vertex candidate
  std::array<float, 9> mWeightMatrix;    // weight matrix
  std::array<float, 3> mVertex;          // cluster centroid posdtion
  std::array<float, 6> mRMS2;            // symmetric matrix: diagonal is RMS2
  float mAvgDistance2;                   // substitute for chi2
#ifdef _ALLOW_DEBUG_TREES_ITS_
  std::vector<Line> mLines;
#endif
};

} // namespace its
} // namespace o2
#endif /* O2_ITSMFT_TRACKING_LINE_H_ */
