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
/// \file ClusterLinesGPU.h
/// \brief GPU-compliant version of ClusterLines, for the moment separated, might create a common traits for ClusterLines + later specifications for each arch, later.
/// \ autrhor: mconcas@cern.ch

#ifndef O2_ITSMFT_TRACKING_CLUSTERLINESGPU_H_
#define O2_ITSMFT_TRACKING_CLUSTERLINESGPU_H_

#include "GPUCommonDef.h"
#include "ITStracking/ClusterLines.h"

namespace o2
{
namespace its
{
namespace GPU
{

struct GPUVertex final {
  GPUhd() GPUVertex()
  {
  }
  GPUhd() GPUVertex(unsigned char isReal) : realVertex{false}
  {
  }
  GPUhd() GPUVertex(float x, float y, float z, float eX, float eY, float eZ) : realVertex{true},
                                                                               xCoord{x},
                                                                               yCoord{y},
                                                                               zCoord{z},
                                                                               errorX{eZ},
                                                                               errorY{eY},
                                                                               errorZ{eZ}
  {
  }
  unsigned char realVertex;
  float xCoord, yCoord, zCoord;
  float errorX, errorY, errorZ;
};

class ClusterLinesGPU final
{
 public:
  GPUd() ClusterLinesGPU(const Line& firstLine, const Line& secondLine); // poor man solution to calculate duplets' centroid
  GPUd() ClusterLinesGPU(const int& firstLabel, const Line& firstLine, const int& secondLabel, const Line& secondLine, const unsigned char weight = false);
  GPUd() void add(const int& lineLabel, const Line& line, const unsigned char weight = false);
  GPUd() void computeClusterCentroid();
  GPUd() float getAvgDistance2() const;
  GPUd() float* getRMS2() const;
  GPUd() inline int getNContributors() { return mNContributors; }
  GPUd() inline float* getVertex() { return mVertex; }

 private:
  float mAMatrix[6];         // AX=B
  float mBMatrix[3];         // AX=B
  int mLabels[2];            // labels
  float mVertexCandidate[3]; // vertex candidate
  float mWeightMatrix[9];    // weight matrix
  float mVertex[3];          // cluster centroid position
  int mNContributors;        // number of participants to this cluster
  float mRMS2[6];            // symmetric matrix: diagonal is RMS2
  float mAvgDistance2;       // substitute for chi2
};

} // namespace GPU
} // namespace its
} // namespace o2
#endif