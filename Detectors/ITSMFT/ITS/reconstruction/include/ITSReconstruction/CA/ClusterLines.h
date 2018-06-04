// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_ITSMFT_RECONSTRUCTION_CA_LINE_H_
#define O2_ITSMFT_RECONSTRUCTION_CA_LINE_H_

#include <array>
#include <vector>

namespace o2
{
namespace ITS
{
namespace CA
{

struct Line final {
  Line();
  Line(std::array<float, 3> firstPoint, std::array<float, 3> secondPoint);

  static float getDistanceFromPoint(const Line& line, const std::array<float, 3> point);
  static std::array<float, 6> getDCAComponents(const Line& line, const std::array<float, 3> point);
  static float getDCA(const Line&, const Line&, const float precision = 1e-14);
  static bool areParallel(const Line&, const Line&, const float precision = 1e-14);

  std::array<float, 3> originPoint, cosinesDirector;
  std::array<float, 6> weightMatrix;
  // weightMatrix is a symmetric matrix internally stored as
  //    0 --> row = 0, col = 0
  //    1 --> 0,1
  //    2 --> 0,2
  //    3 --> 1,1
  //    4 --> 1,2
  //    5 --> 2,2
};

class ClusterLines final
{
 public:
  ClusterLines(const int firstLabel, const Line& firstLine, const int secondLabel, const Line& secondLine,
               const bool weight = false);
  void add(const int lineLabel, const Line& line, const bool weight = false);
  void computeClusterCentroid();
  float getAvgDistance2() const;
  std::array<float, 6> getRMS2() const;
  inline std::vector<int> getLabels() const { return mLabels; };
  inline int getSize() const { return mLabels.size(); };
  inline std::array<float, 3> getVertex() const { return mVertex; }
  std::vector<Line> mLines;

 protected:
  std::array<float, 6> mAMatrix;         // AX=B
  std::array<float, 3> mBMatrix;         // AX=B
  std::vector<int> mLabels;              // labels
  std::array<float, 3> mVertexCandidate; // vertex candidate
  std::array<float, 9> mWeightMatrix;    // weight matrix
  std::array<float, 3> mVertex;          // cluster centroid position
};

} // namespace CA
} // namespace ITS
} // namespace o2
#endif /* O2_ITSMFT_RECONSTRUCTION_CA_LINE_H_ */