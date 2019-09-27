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
/// \file StandaloneDebugger.cxx
/// \brief
/// \author matteo.concas@cern.ch

#include <string>
#include "ITStracking/Cluster.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/ClusterLines.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/StandaloneDebugger.h"

namespace o2
{
namespace its
{

StandaloneDebugger::StandaloneDebugger(const std::string debugTreeFileName)
{
  mDebugTreeFileName = debugTreeFileName;
  mTreeStream = new o2::utils::TreeStreamRedirector(debugTreeFileName.data(), "recreate");
}

StandaloneDebugger::~StandaloneDebugger()
{
  delete mTreeStream;
}

void StandaloneDebugger::fillCombinatoricsTree(std::vector<Tracklet> comb01, std::vector<Tracklet> comb12)
{
  for (auto& combination : comb01) {
    (*mTreeStream)
      << "combinatorics01"
      << "tanLambda=" << combination.tanLambda
      << "phi=" << combination.phiCoordinate
      << "\n";
  }

  for (auto& combination : comb12) {
    (*mTreeStream)
      << "combinatorics12"
      << "tanLambda=" << combination.tanLambda
      << "phi=" << combination.phiCoordinate
      << "\n";
  }
}

void StandaloneDebugger::fillCombinatoricsMCTree(std::vector<Tracklet> comb01, std::vector<Tracklet> comb12)
{
  mTreeStream->GetDirectory()->cd(); // in case of existing other open files
  for (auto& combination : comb01) {
    (*mTreeStream)
      << "combinatorics01_MC"
      << "tanLambda=" << combination.tanLambda
      << "phi=" << combination.phiCoordinate
      << "\n";
  }

  for (auto& combination : comb12) {
    (*mTreeStream)
      << "combinatorics12_MC"
      << "tanLambda=" << combination.tanLambda
      << "phi=" << combination.phiCoordinate
      << "\n";
  }
}

void StandaloneDebugger::fillTrackletSelectionTree(std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
                                                   std::vector<Tracklet> comb01,
                                                   std::vector<Tracklet> comb12,
                                                   std::vector<std::array<int, 2>> allowedTracklets,
                                                   const ROframe* event)
{
  assert(event != nullptr);
  int id = event->getROFrameId();
  for (auto& trackletPair : allowedTracklets) {
    o2::MCCompLabel lblClus0 = event->getClusterLabels(0, clusters[0][comb01[trackletPair[0]].firstClusterIndex].clusterId);
    o2::MCCompLabel lblClus1 = event->getClusterLabels(1, clusters[1][comb01[trackletPair[0]].secondClusterIndex].clusterId);
    o2::MCCompLabel lblClus2 = event->getClusterLabels(2, clusters[2][comb12[trackletPair[1]].secondClusterIndex].clusterId);
    unsigned char isValidated{(lblClus0.compare(lblClus1) == 1 && lblClus0.compare(lblClus2) == 1)};
    float deltaPhi{gpu::GPUCommonMath::Abs(comb01[trackletPair[0]].phiCoordinate - comb12[trackletPair[1]].phiCoordinate)};
    float deltaTanLambda{gpu::GPUCommonMath::Abs(comb01[trackletPair[0]].tanLambda - comb12[trackletPair[1]].tanLambda)};
    mTreeStream->GetDirectory()->cd(); // in case of existing other open files
    (*mTreeStream)
      << "selectedTracklets"
      << "ROframeId=" << id
      << "deltaTanlambda=" << deltaTanLambda
      << "deltaPhi=" << deltaPhi
      << "isValidated=" << isValidated
      << "cluster0z=" << clusters[0][comb01[trackletPair[0]].firstClusterIndex].zCoordinate
      << "cluster0r=" << clusters[0][comb01[trackletPair[0]].firstClusterIndex].rCoordinate
      << "cluster1z=" << clusters[1][comb01[trackletPair[0]].secondClusterIndex].zCoordinate
      << "cluster1r=" << clusters[1][comb01[trackletPair[0]].secondClusterIndex].rCoordinate
      << "cluster2z=" << clusters[2][comb12[trackletPair[1]].secondClusterIndex].zCoordinate
      << "cluster2r=" << clusters[2][comb12[trackletPair[1]].secondClusterIndex].rCoordinate
      << "lblClus0=" << lblClus0
      << "lblClus1=" << lblClus1
      << "lblClus2=" << lblClus2
      << "\n";
  }
}

void StandaloneDebugger::fillStridedTrackletSelectionTree(std::array<std::vector<Cluster>, constants::its::LayersNumberVertexer>& clusters,
                                                          std::vector<Tracklet> comb01,
                                                          std::vector<Tracklet> comb12,
                                                          std::vector<std::array<int, 2>> allowedTracklets,
                                                          std::vector<int> nFoundSelections,
                                                          const int stride,
                                                          const ROframe* event)
{
  assert(event != nullptr);
  int id = event->getROFrameId();
  for (int iClusterIndex{0}; iClusterIndex < clusters[1].size(); ++iClusterIndex) {
    const int stridedIndex{iClusterIndex * stride};
    for (int iTrackletPair{0}; iTrackletPair < nFoundSelections[iClusterIndex]; ++iTrackletPair) {
      auto& trackletPair = allowedTracklets[stridedIndex + iTrackletPair];
      o2::MCCompLabel lblClus0 = event->getClusterLabels(0, clusters[0][comb01[trackletPair[0]].firstClusterIndex].clusterId);
      o2::MCCompLabel lblClus1 = event->getClusterLabels(1, clusters[1][comb01[trackletPair[0]].secondClusterIndex].clusterId);
      o2::MCCompLabel lblClus2 = event->getClusterLabels(2, clusters[2][comb12[trackletPair[1]].secondClusterIndex].clusterId);
      unsigned char isValidated{(lblClus0.compare(lblClus1) == 1 && lblClus0.compare(lblClus2) == 1)};
      float deltaTanLambda{gpu::GPUCommonMath::Abs(comb01[trackletPair[0]].tanLambda - comb12[trackletPair[1]].tanLambda)};
      mTreeStream->GetDirectory()->cd(); // in case of existing other open files
      (*mTreeStream)
        << "selectedTracklets"
        << "ROframeId=" << id
        << "deltaTanlambda=" << deltaTanLambda
        << "isValidated=" << isValidated
        << "cluster0z=" << clusters[0][comb01[trackletPair[0]].firstClusterIndex].zCoordinate
        << "cluster0r=" << clusters[0][comb01[trackletPair[0]].firstClusterIndex].rCoordinate
        << "cluster1z=" << clusters[1][comb01[trackletPair[0]].secondClusterIndex].zCoordinate
        << "cluster1r=" << clusters[1][comb01[trackletPair[0]].secondClusterIndex].rCoordinate
        << "cluster2z=" << clusters[2][comb12[trackletPair[1]].secondClusterIndex].zCoordinate
        << "cluster2r=" << clusters[2][comb12[trackletPair[1]].secondClusterIndex].rCoordinate
        << "lblClus0=" << lblClus0
        << "lblClus1=" << lblClus1
        << "lblClus2=" << lblClus2
        << "\n";
    }
  }
}

void StandaloneDebugger::fillLinesSummaryTree(std::vector<Line> lines, const ROframe* event)
{
  assert(event != nullptr);
  int id = event->getROFrameId();
  const o2::its::Line zAxis{std::array<float, 3>{0.f, 0.f, -1.f}, std::array<float, 3>{0.f, 0.f, 1.f}};
  for (auto& tracklet : lines) {
    float dcaz = Line::getDCA(tracklet, zAxis);
    (*mTreeStream)
      << "linesSummary"
      << "ROframeId=" << id
      << "oX=" << tracklet.originPoint[0]
      << "oY=" << tracklet.originPoint[1]
      << "oZ=" << tracklet.originPoint[2]
      << "c1=" << tracklet.cosinesDirector[0]
      << "c2=" << tracklet.cosinesDirector[1]
      << "c2=" << tracklet.cosinesDirector[2]
      << "DCAZaxis=" << dcaz
      // TODO: Line::getDistanceFromPoint(line, MC_vertex)
      << "\n";
  }
}

void StandaloneDebugger::fillPairsInfoTree(std::vector<Line> lines, const ROframe* event)
{
  assert(event != nullptr);
  int id = event->getROFrameId();
  for (unsigned int iLine1{0}; iLine1 < lines.size(); ++iLine1) { // compute centroids for every line pair
    auto line1 = lines[iLine1];
    for (unsigned int iLine2{iLine1 + 1}; iLine2 < lines.size(); ++iLine2) {
      auto line2 = lines[iLine2];
      ClusterLines cluster{-1, line1, -1, line2};
      auto vtx = cluster.getVertex();
      if (std::hypot(vtx[0], vtx[1]) < 1.98 * 1.98) {
        float dcaPair = Line::getDCA(line1, line2);
        (*mTreeStream)
          << "pairInfo"
          << "centroids"
          << "ROframeId=" << id
          << "xCoord=" << vtx[0]
          << "yCoord=" << vtx[1]
          << "zCoord=" << vtx[2]
          << "DCApair=" << dcaPair
          << "\n";
      }
    }
    // TODO: get primary vertex montecarlo position
    // mLinesData.push_back(Line::getDCAComponents(line1, std::array<float, 3>{0., 0., 0.}));
  }
}

} // namespace its
} // namespace o2