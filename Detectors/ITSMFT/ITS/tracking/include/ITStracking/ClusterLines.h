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
#include <Math/SMatrix.h>
#include <Math/SVector.h>
#include "ITStracking/Cluster.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Tracklet.h"
#include "GPUCommonRtypes.h"

namespace o2::its
{

struct Line final {
#if !defined(__HIPCC__) && !defined(__CUDACC__) // hide the class completely for gpu-cc
  using SVector3f = ROOT::Math::SVector<float, 3>;
  using SMatrix3f = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;

  Line() = default;
  Line(const Tracklet&, const Cluster*, const Cluster*);
  bool operator==(const Line&) const = default;

  static float getDistance2FromPoint(const Line& line, const std::array<float, 3>& point);
  static float getDistanceFromPoint(const Line& line, const std::array<float, 3>& point);
  static SMatrix3f getDCAComponents(const Line& line, const std::array<float, 3>& point);
  static float getDCA2(const Line&, const Line&, const float precision = constants::Tolerance);
  static float getDCA(const Line&, const Line&, const float precision = constants::Tolerance);
  bool isEmpty() const noexcept;
  void print() const;

  SVector3f originPoint;
  SVector3f cosinesDirector;
  TimeEstBC mTime;

  ClassDefNV(Line, 1);
#endif
};

class ClusterLines final
{
#if !defined(__HIPCC__) && !defined(__CUDACC__) // hide the class completely for gpu-cc
  using SMatrix3 = ROOT::Math::SMatrix<double, 3, 3, ROOT::Math::MatRepSym<double, 3>>;
  using SMatrix3f = ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3>>;
  using SVector3 = ROOT::Math::SVector<double, 3>;

 public:
  ClusterLines() = default;
  ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine);
  ClusterLines(gsl::span<const int> lineLabels, gsl::span<const Line> lines);
  void add(const int lineLabel, const Line& line);
  void computeClusterCentroid();
  void accumulate(const Line& line);
  bool isValid() const noexcept { return mIsValid; }
  auto const& getVertex() const { return mVertex; }
  const float* getRMS2() const { return mRMS2.Array(); }
  float getAvgDistance2() const { return mAvgDistance2; }
  auto getSize() const noexcept { return mLabels.size(); }
  auto& getLabels() const noexcept { return mLabels; }
  const auto& getTimeStamp() const noexcept { return mTime; }
  bool operator==(const ClusterLines& rhs) const noexcept;
  float getR2() const noexcept { return (mVertex[0] * mVertex[0]) + (mVertex[1] * mVertex[1]); }
  float getR() const noexcept { return std::sqrt(getR2()); }

 protected:
  SMatrix3 mAMatrix;                 // AX=B, symmetric normal matrix
  SVector3 mBMatrix;                 // AX=B, right-hand side
  std::array<float, 3> mVertex = {}; // cluster centroid position
  SMatrix3f mRMS2;                   // symmetric matrix: diagonal is RMS2
  float mAvgDistance2 = 0.f;         // substitute for chi2
  bool mIsValid = false;             // true if linear system was solved successfully
  TimeEstBC mTime;                   // time stamp
  std::vector<int> mLabels;          // contributing labels

  ClassDefNV(ClusterLines, 1);
#endif
};

} // namespace o2::its
#endif /* O2_ITS_CLUSTERLINES_H */
