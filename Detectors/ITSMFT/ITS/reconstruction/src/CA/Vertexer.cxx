// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <algorithm>
#include <iostream>
#include <cmath>
#include <limits>
#include <iomanip>
#include <chrono>
#include <utility>
#include <tuple>

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/Layer.h"
#include "ITSReconstruction/CA/ClusterLines.h"
#include "ITSReconstruction/CA/Vertexer.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

using Constants::IndexTable::PhiBins;
using Constants::IndexTable::ZBins;
using Constants::ITS::LayersRCoordinate;
using Constants::ITS::LayersZCoordinate;
using Constants::Math::TwoPi;
using IndexTableUtils::getZBinIndex;

Vertexer::Vertexer(const Event& event) : mEvent{ event }, mAverageClustersRadii{ std::array<float, 3>{ 0.f, 0.f, 0.f } }
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    const Layer& currentLayer{ event.getLayer(iLayer) };
    const int clustersNum{ currentLayer.getClustersSize() };
    mClusters[iLayer].clear();
    if (clustersNum > mClusters[iLayer].capacity()) {
      mClusters[iLayer].reserve(clustersNum);
    }
    for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
      mClusters[iLayer].emplace_back(iLayer, currentLayer.getCluster(iCluster));
    }
    if (mClusters[iLayer].size() != 0) {
      const float inverseNumberOfClusters{ 1.f / mClusters[iLayer].size() };
      for (auto& cluster : mClusters[iLayer]) {
        mAverageClustersRadii[iLayer] += cluster.rCoordinate;
      }
      mAverageClustersRadii[iLayer] *= inverseNumberOfClusters;
    } else {
      mAverageClustersRadii[iLayer] = LayersRCoordinate()[iLayer];
    }
  }
  mDeltaRadii10 = mAverageClustersRadii[1] - mAverageClustersRadii[0];
  mDeltaRadii21 = mAverageClustersRadii[2] - mAverageClustersRadii[1];
  mMaxDirectorCosine3 =
    LayersZCoordinate()[2] / std::sqrt(LayersZCoordinate()[2] * LayersZCoordinate()[2] +
                                       (mDeltaRadii10 + mDeltaRadii21) * (mDeltaRadii10 + mDeltaRadii21));
}

Vertexer::~Vertexer(){};

void Vertexer::initialise(const float zCut, const float phiCut, const float pairCut, const float clusterCut,
                          const int clusterContributorsCut)
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });
  }
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    const int clustersNum = static_cast<int>(mClusters[iLayer].size());
    int previousBinIndex{ 0 };
    mIndexTables[iLayer][0] = 0;
    for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
      const int currentBinIndex{ mClusters[iLayer][iCluster].indexTableBinIndex };
      if (currentBinIndex > previousBinIndex) {
        for (int iBin{ previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
          mIndexTables[iLayer][iBin] = iCluster;
        }
        previousBinIndex = currentBinIndex;
      }
    }
    for (int iBin{ previousBinIndex + 1 }; iBin <= ZBins * PhiBins; iBin++) {
      mIndexTables[iLayer][iBin] = clustersNum;
    }
  }

  mZCut = zCut > Constants::ITS::LayersZCoordinate()[0] ? LayersZCoordinate()[0] : zCut;
  mPhiCut = phiCut > TwoPi ? TwoPi : phiCut;
  mPairCut = pairCut;
  mClusterCut = clusterCut;
  mClusterContributorsCut = clusterContributorsCut;
  mPhiSpan = static_cast<int>(std::ceil(PhiBins * mPhiCut / TwoPi));
  mZSpan = static_cast<int>(std::ceil(mZCut * Constants::IndexTable::InverseZBinSize()[0]));
  mVertexerInitialised = true;
}

void Vertexer::initialise(const std::tuple<float, float, float, float, int> initParams)
{
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });
  }
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    const int clustersNum = static_cast<int>(mClusters[iLayer].size());
    int previousBinIndex{ 0 };
    mIndexTables[iLayer][0] = 0;
    for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
      const int currentBinIndex{ mClusters[iLayer][iCluster].indexTableBinIndex };
      if (currentBinIndex > previousBinIndex) {
        for (int iBin{ previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
          mIndexTables[iLayer][iBin] = iCluster;
        }
        previousBinIndex = currentBinIndex;
      }
    }
    for (int iBin{ previousBinIndex + 1 }; iBin <= ZBins * PhiBins; iBin++) {
      mIndexTables[iLayer][iBin] = clustersNum;
    }
  }

  mZCut =
    std::get<0>(initParams) > Constants::ITS::LayersZCoordinate()[0] ? LayersZCoordinate()[0] : std::get<0>(initParams);
  mPhiCut = std::get<1>(initParams) > TwoPi ? TwoPi : std::get<1>(initParams);
  mPairCut = std::get<2>(initParams);
  mClusterCut = std::get<3>(initParams);
  mClusterContributorsCut = std::get<4>(initParams);
  mPhiSpan = static_cast<int>(std::ceil(PhiBins * mPhiCut / TwoPi));
  mZSpan = static_cast<int>(std::ceil(mZCut * Constants::IndexTable::InverseZBinSize()[0]));
  mVertexerInitialised = true;
}

const std::vector<std::pair<int, int>> Vertexer::selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
                                                                const std::array<int, 4>& selectedBinsRect)
{
  std::vector<std::pair<int, int>> filteredBins{};
  int phiBinsNum{ selectedBinsRect[3] - selectedBinsRect[1] + 1 };
  if (phiBinsNum < 0)
    phiBinsNum += PhiBins;
  filteredBins.reserve(phiBinsNum);
  for (int iPhiBin{ selectedBinsRect[1] }, iPhiCount{ 0 }; iPhiCount < phiBinsNum;
       iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
    const int firstBinIndex{ IndexTableUtils::getBinIndex(selectedBinsRect[0], iPhiBin) };
    filteredBins.emplace_back(
      indexTable[firstBinIndex],
      IndexTableUtils::countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }
  return filteredBins;
}

void Vertexer::findTracklets(const bool useMCLabel)
{
  if (mVertexerInitialised) {
    std::vector<std::pair<int, int>> clusters0, clusters2;
    std::vector<bool> usedCluster2Flags, usedCluster0Flags;
    usedCluster2Flags.resize(mClusters[2].size(), false);
    usedCluster0Flags.resize(mClusters[0].size(), false);

    for (int iBin1{ 0 }; iBin1 < ZBins * PhiBins; ++iBin1) {

      int ZBinLow0{ std::max(0, ZBins - static_cast<int>(std::ceil((ZBins - iBin1 % ZBins + 1) *
                                                                   (mDeltaRadii21 + mDeltaRadii10) / mDeltaRadii10))) };
      int ZBinHigh0{ std::min(
        static_cast<int>(std::ceil((iBin1 % ZBins + 1) * (mDeltaRadii21 + mDeltaRadii10) / mDeltaRadii21)),
        ZBins - 1) };
      // int ZBinLow0 { 0 };
      // int ZBinHigh0 { ZBins -1 };

      int PhiBin1{ static_cast<int>(iBin1 / ZBins) };
      clusters0 = selectClusters(
        mIndexTables[0],
        std::array<int, 4>{ ZBinLow0, (PhiBin1 - mPhiSpan < 0) ? PhiBins + (PhiBin1 - mPhiSpan) : PhiBin1 - mPhiSpan,
                            ZBinHigh0,
                            (PhiBin1 + mPhiSpan > PhiBins) ? PhiBin1 + mPhiSpan - PhiBins : PhiBin1 + mPhiSpan });

      for (int iCluster1{ mIndexTables[1][iBin1] }; iCluster1 < mIndexTables[1][iBin1 + 1]; ++iCluster1) {
        bool trackFound = false;
        for (int iRow0{ 0 }; iRow0 < clusters0.size(); ++iRow0) {
          for (int iCluster0{ std::get<0>(clusters0[iRow0]) };
               iCluster0 < std::get<0>(clusters0[iRow0]) + std::get<1>(clusters0[iRow0]); ++iCluster0) {
            if (usedCluster0Flags[iCluster0])
              continue;
            if (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut ||
                std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) >
                  TwoPi - mPhiCut) {
              float ZProjection{ mClusters[0][iCluster0].zCoordinate +
                                 (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) *
                                   (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                   (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              float ZProjectionInner{ mClusters[0][iCluster0].zCoordinate +
                                      (mClusters[0][iCluster0].rCoordinate) *
                                        (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                        (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              if (std::abs(ZProjection) > (LayersZCoordinate()[0] + mZCut))
                continue;
              int ZProjectionBin{ (ZProjection < -LayersZCoordinate()[0])
                                    ? 0
                                    : (ZProjection > LayersZCoordinate()[0]) ? ZBins - 1
                                                                             : getZBinIndex(2, ZProjection) };
              int ZBinLow2{ (ZProjectionBin - mZSpan < 0) ? 0 : ZProjectionBin - mZSpan };
              int ZBinHigh2{ (ZProjectionBin + mZSpan > ZBins - 1) ? ZBins - 1 : ZProjectionBin + mZSpan };
              // int ZBinLow2  { 0 };
              // int ZBinHigh2  { ZBins - 1 };
              int PhiBinLow2{ (PhiBin1 - mPhiSpan < 0) ? PhiBins + PhiBin1 - mPhiSpan : PhiBin1 - mPhiSpan };
              int PhiBinHigh2{ (PhiBin1 + mPhiSpan > PhiBins - 1) ? PhiBin1 + mPhiSpan - PhiBins : PhiBin1 + mPhiSpan };
              // int PhiBinLow2{ 0 };
              // int PhiBinHigh2{ PhiBins - 1 };
              clusters2 =
                selectClusters(mIndexTables[2], std::array<int, 4>{ ZBinLow2, PhiBinLow2, ZBinHigh2, PhiBinHigh2 });
              for (int iRow2{ 0 }; iRow2 < clusters2.size(); ++iRow2) {
                for (int iCluster2{ std::get<0>(clusters2[iRow2]) };
                     iCluster2 < std::get<0>(clusters2[iRow2]) + std::get<1>(clusters2[iRow2]); ++iCluster2) {
                  if (usedCluster2Flags[iCluster2])
                    continue;
                  float ZProjectionRefined{
                    mClusters[0][iCluster0].zCoordinate +
                    (mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate) *
                      (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                      (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate)
                  };
                  bool testMC{ !useMCLabel ||
                               (mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() ==
                                  mEvent.getLayer(2).getClusterLabel(mClusters[2][iCluster2].clusterId).getTrackID() &&
                                mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() ==
                                  mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()) };
                  float absDeltaPhi{ std::abs(mClusters[2][iCluster2].phiCoordinate -
                                              mClusters[1][iCluster1].phiCoordinate) };
                  float absDeltaZ{ std::abs(mClusters[2][iCluster2].zCoordinate - ZProjectionRefined) };
                  if (absDeltaZ < mZCut && (absDeltaPhi < mPhiCut || std::abs(absDeltaPhi - TwoPi) < mPhiCut && testMC)) {
                    mTracklets.emplace_back(Line{
                      std::array<float, 3>{ mClusters[0][iCluster0].xCoordinate, mClusters[0][iCluster0].yCoordinate,
                                            mClusters[0][iCluster0].zCoordinate },
                      std::array<float, 3>{ mClusters[1][iCluster1].xCoordinate, mClusters[1][iCluster1].yCoordinate,
                                            mClusters[1][iCluster1].zCoordinate } });
                    if (std::abs(mTracklets.back().cosinesDirector[2]) < mMaxDirectorCosine3) {
                      usedCluster0Flags[iCluster0] = true;
                      usedCluster2Flags[iCluster2] = true;
                      trackFound = true;
                      break;
                    } else {
                      mTracklets.pop_back();
                    }
                  }
                }
                if (trackFound)
                  break;
              }
            }
            if (trackFound)
              break;
          }
          if (trackFound)
            break;
        }
      }
    }
    mTrackletsFound = true;
  }
}

void Vertexer::findVertices()
{
  if (mTrackletsFound) {
    const int numTracklets{ static_cast<int>(mTracklets.size()) };
    std::vector<bool> usedTracklets{};
    usedTracklets.resize(mTracklets.size(), false);
    for (int tracklet1{ 0 }; tracklet1 < numTracklets; ++tracklet1) {
      if (usedTracklets[tracklet1])
        continue;
      for (int tracklet2{ tracklet1 + 1 }; tracklet2 < numTracklets; ++tracklet2) {
        if (usedTracklets[tracklet2])
          continue;
        if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mPairCut) {
          mTrackletClusters.emplace_back(tracklet1, mTracklets[tracklet1], tracklet2, mTracklets[tracklet2]);
          std::array<float, 3> tmpVertex{ mTrackletClusters.back().getVertex() };
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
            mTrackletClusters.pop_back();
            break;
          }
          usedTracklets[tracklet1] = true;
          usedTracklets[tracklet2] = true;
          for (int tracklet3{ 0 }; tracklet3 < numTracklets; ++tracklet3) {
            if (usedTracklets[tracklet3])
              continue;
            if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mPairCut) {
              mTrackletClusters.back().add(tracklet3, mTracklets[tracklet3]);
              usedTracklets[tracklet3] = true;
              tmpVertex = mTrackletClusters.back().getVertex();
            }
          }
          break;
        }
      }
    }
    std::sort(mTrackletClusters.begin(), mTrackletClusters.end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    int noClusters{ static_cast<int>(mTrackletClusters.size()) };
    for (int iCluster1{ 0 }; iCluster1 < noClusters; ++iCluster1) {
      std::array<float, 3> vertex1{ mTrackletClusters[iCluster1].getVertex() };
      std::array<float, 3> vertex2{};
      for (int iCluster2{ iCluster1 + 1 }; iCluster2 < noClusters; ++iCluster2) {
        vertex2 = mTrackletClusters[iCluster2].getVertex();
        if (std::abs(vertex1[2] - vertex2[2]) < mClusterCut) {

          float distance{ (vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                          (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                          (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]) };
          if (distance <= mPairCut * mPairCut) {
            for (auto label : mTrackletClusters[iCluster2].getLabels()) {
              mTrackletClusters[iCluster1].add(label, mTracklets[label]);
              vertex1 = mTrackletClusters[iCluster1].getVertex();
            }
          }
          mTrackletClusters.erase(mTrackletClusters.begin() + iCluster2);
          --iCluster2;
          --noClusters;
        }
      }
    }
    for (int iCluster{ 0 }; iCluster < noClusters; ++iCluster) {
      if (mTrackletClusters[iCluster].getSize() < mClusterContributorsCut && noClusters > 1) {
        mTrackletClusters.erase(mTrackletClusters.begin() + iCluster);
        noClusters--;
        continue;
      }
      float dist{ 0. };
      for (auto& line : mTrackletClusters[iCluster].mLines) {
        dist += Line::getDistanceFromPoint(line, mTrackletClusters[iCluster].getVertex()) /
                mTrackletClusters[iCluster].getSize();
      }
      if (mTrackletClusters[iCluster].getVertex()[0] * mTrackletClusters[iCluster].getVertex()[0] +
            mTrackletClusters[iCluster].getVertex()[1] * mTrackletClusters[iCluster].getVertex()[1] <
          1.98 * 1.98) {
        // #ifdef DEBUG_BUILD
        //         mLegacyVertices.emplace_back(
        //           std::make_tuple(mTrackletClusters[iCluster].getVertex(), mTrackletClusters[iCluster].getSize(),
        //           dist));
        // #else
        //         mLegacyVertices.emplace_back(mTrackletClusters[iCluster].getVertex());
        // #endif
        mVertices.emplace_back(
          Point3D<float>{ mTrackletClusters[iCluster].getVertex()[0], mTrackletClusters[iCluster].getVertex()[1],
                          mTrackletClusters[iCluster].getVertex()[2] },
          mTrackletClusters[iCluster].getRMS2(),        // Symm matrix. Diagonal: RMS2 components,
                                                        // off-diagonal: square mean of projections on planes.
          mTrackletClusters[iCluster].getSize(),        // Contributors
          mTrackletClusters[iCluster].getAvgDistance2() // In place of chi2
          );
        mVertices.back().setTimeStamp(mROFrame);
      }
    }
  }
}

#ifdef DEBUG_BUILD
void Vertexer::printIndexTables()
{
  for (int iTables{ 0 }; iTables < Constants::ITS::LayersNumberVertexer; ++iTables) {
    std::cout << "Table " << iTables << std::endl;
    for (int iIndexPhi{ 0 }; iIndexPhi < PhiBins; ++iIndexPhi) {
      for (int iIndexZeta{ 0 }; iIndexZeta < ZBins; ++iIndexZeta) {
        std::cout << mIndexTables[iTables][iIndexZeta + ZBins * iIndexPhi] << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << mIndexTables[iTables][ZBins * PhiBins] << "\t";
  }
}

void Vertexer::dumpTracklets()
{
  for (auto& cluster : mClusters[0]) {
    if (mEvent.getLayer(0).getClusterLabel(cluster.clusterId).getTrackID() == 60) {
      std::cout << "x: " << cluster.xCoordinate << " y: " << cluster.yCoordinate << " z: " << cluster.zCoordinate
                << std::endl;
    }
  }
  for (auto& tracklet : mTracklets) {
    std::cout << "or id: " << tracklet.originID << "\t de id: " << tracklet.destinID << std::endl;
  }
}
#endif

} // namespace CA
} // namespace ITS
} // namespace o2