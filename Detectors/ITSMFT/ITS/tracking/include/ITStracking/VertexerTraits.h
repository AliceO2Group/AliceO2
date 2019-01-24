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
/// \file VertexerTraits.h
/// \brief
///

#ifndef O2_ITS_TRACKING_VERTEXER_TRAITS_H_
#define O2_ITS_TRACKING_VERTEXER_TRAITS_H_

#include "ITStracking/Configuration.h"
#include <array>
#include <vector>

#include "ITStracking/Cluster.h"
#include "ClusterLines.h"

namespace o2
{

namespace ITS
{

class ROframe;

using Constants::IndexTable::PhiBins;
using Constants::IndexTable::ZBins;
using Constants::ITS::LayersNumberVertexer;

struct lightVertex {
  lightVertex(float x, float y, float z, std::array<float, 6> rms2, float avgdis2, int cont, int stamp);
  float mX;
  float mY;
  float mZ;
  std::array<float, 6> mRMS2;
  float mAvgDistance2;
  int mContributors;
  int mTimeStamp;
};

inline lightVertex::lightVertex(float x, float y, float z, std::array<float, 6> rms2, float avgdis2, int cont, int stamp) : mX(x), mY(y), mZ(z), mRMS2(rms2), mAvgDistance2(avgdis2), mContributors(cont), mTimeStamp(stamp)
{
}

class VertexerTraits
{
 public:
  VertexerTraits();
  virtual ~VertexerTraits() = default;

  // virtual vertexer interface
  virtual void reset();
  virtual void initialise(ROframe*);
  virtual void computeTracklets(const bool useMCLabel = false);
  virtual void computeVertices();

  void updateVertexingParameters(const VertexingParameters& vrtPar);
  static const std::vector<std::pair<int, int>> selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                               const std::array<int, 4>& selectedBinsRect);
  std::vector<lightVertex> getVertices() const { return mVertices; }
  void dumpVertexerTraits();

 protected:
  VertexingParameters mVrtParams;
  std::array<std::array<int, ZBins * PhiBins + 1>, LayersNumberVertexer> mIndexTables;
  std::vector<lightVertex> mVertices;

  // Frame related quantities
  o2::ITS::ROframe* mEvent;
  uint32_t mROframe;
  std::array<std::vector<Cluster>, Constants::ITS::LayersNumberVertexer> mClusters;
  std::vector<Line> mTracklets;
  std::array<float, 3> mAverageClustersRadii;
  float mDeltaRadii10, mDeltaRadii21;
  float mMaxDirectorCosine3;
  std::vector<ClusterLines> mTrackletClusters;
};

inline void VertexerTraits::updateVertexingParameters(const VertexingParameters& vrtPar)
{
  mVrtParams = vrtPar;
}

} // namespace ITS
} // namespace o2
#endif /* O2_ITS_TRACKING_VERTEXER_TRAITS_H_ */
