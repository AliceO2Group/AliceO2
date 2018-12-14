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

#ifndef VERTEXERTRAITS_H_
#define VERTEXERTRAITS_H_

#include "ITStracking/Configuration.h"
#include <array>
#include <vector>

#include "ITStracking/Cluster.h"
#include "ITStracking/ROframe.h"
#include "ClusterLines.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace ITS
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;
using Constants::IndexTable::PhiBins;
using Constants::IndexTable::ZBins;
using Constants::ITS::LayersNumberVertexer;

class VertexerTraits
{
 public:
  VertexerTraits();
  virtual ~VertexerTraits() = default;

  // virtual vertexer interface
  virtual void reset();
  virtual void initialise(const ROframe&);
  virtual void computeTracklets(const bool useMCLabel = false);
  virtual void computeVertices();

  void updateVertexingParameters(const VertexingParameters& vrtPar);
  std::vector<Vertex> exportVertices() { return mVertices; }

  static const std::vector<std::pair<int, int>> selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                               const std::array<int, 4>& selectedBinsRect);
  std::vector<Vertex> getVertices() const { return mVertices; }
  void dumpVertexerTraits();

 protected:
  VertexingParameters mVrtParams;
  std::array<std::array<int, ZBins * PhiBins + 1>, LayersNumberVertexer> mIndexTables;
  std::vector<Vertex> mVertices;

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
#endif /* VERTEXERTRAITS_H_ */