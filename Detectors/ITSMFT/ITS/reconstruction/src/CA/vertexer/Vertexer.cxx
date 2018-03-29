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

#include "ITSReconstruction/CA/Constants.h"
#include "ITSReconstruction/CA/Cluster.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/Layer.h"
#include "ITSReconstruction/CA/vertexer/ClusterLines.h"
#include "ITSReconstruction/CA/vertexer/Vertexer.h"
#include "ITSReconstruction/CA/IndexTableUtils.h"

namespace o2
{
namespace ITS
{
namespace CA
{

using Constants::ITS::LayersRCoordinate;
using Constants::ITS::LayersZCoordinate;
using Constants::IndexTable::PhiBins;
using Constants::IndexTable::ZBins;
using Constants::Math::TwoPi;
using IndexTableUtils::getZBinIndex;

Vertexer::Vertexer(const Event& event) : mEvent{ event },
mAverageClustersRadii{ std::array<float, 3>{ 0.f, 0.f, 0.f } }
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
    const float inverseNumberOfClusters { 1.f/mClusters[iLayer].size() };
    for ( auto& cluster : mClusters[iLayer] ) {
      mAverageClustersRadii[iLayer] += cluster.rCoordinate;
    }
    mAverageClustersRadii[iLayer] *= inverseNumberOfClusters;
  }
  mDeltaRadii10 = mAverageClustersRadii[1] - mAverageClustersRadii[0];
  mDeltaRadii21 = mAverageClustersRadii[2] - mAverageClustersRadii[1];
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

void Vertexer::printIndexTables() // For debug only purposes, TBR
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
    filteredBins.emplace_back( indexTable[firstBinIndex],
      IndexTableUtils::countRowSelectedBins(indexTable, iPhiBin, selectedBinsRect[0], selectedBinsRect[2]));
  }
  return filteredBins;
}

void Vertexer::findTracklets()
{
  if (mVertexerInitialised) {
    // std::chrono::time_point<std::chrono::system_clock> start, end;
    // start = std::chrono::system_clock::now();

    std::vector<std::pair<int, int>> clusters0;
    std::vector<std::pair<int, int>> clusters2;

    for (int iBin1{ 0 }; iBin1 < ZBins * PhiBins; ++iBin1) {

      int ZBinLow0{ std::max(0, ZBins - static_cast<int>(std::ceil((ZBins - iBin1 % ZBins + 1) *
                                                                    (mDeltaRadii21 + mDeltaRadii10) / mDeltaRadii10))) };
      int ZBinHigh0{ std::min( static_cast<int>(std::ceil((iBin1 % ZBins + 1) * (mDeltaRadii21 + mDeltaRadii10) / mDeltaRadii21)),
        ZBins - 1) };
      // int ZBinLow0 { 0 };
      // int ZBinHigh0 { ZBins -1 };

      int PhiBin1{ static_cast<int>(iBin1 / ZBins) };

      clusters0 = selectClusters( mIndexTables[0], 
        std::array<int, 4>{ ZBinLow0, (PhiBin1 - mPhiSpan < 0) ? PhiBins + (PhiBin1 - mPhiSpan) : PhiBin1 - mPhiSpan, 
        ZBinHigh0, (PhiBin1 + mPhiSpan > PhiBins) ? PhiBin1 + mPhiSpan - PhiBins : PhiBin1 + mPhiSpan });

      for (int iCluster1{ mIndexTables[1][iBin1] }; iCluster1 < mIndexTables[1][iBin1 + 1]; ++iCluster1) {
        for (int iRow0{ 0 }; iRow0 < clusters0.size(); ++iRow0) {
          for (int iCluster0{ std::get<0>(clusters0[iRow0]) }; iCluster0 < std::get<0>(clusters0[iRow0]) + std::get<1>(clusters0[iRow0]); ++iCluster0) {

            if (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) > TwoPi - mPhiCut) {

              float ZProjection{ mClusters[0][iCluster0].zCoordinate +  (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              float ZProjectionInner{ mClusters[0][iCluster0].zCoordinate +  (mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };

              // if (std::abs(ZProjection) > (LayersZCoordinate()[0] + mZCut)) Apparently vertex can be outside the ITS
              //   continue;

              int ZProjectionBin{ (ZProjection < -LayersZCoordinate()[0]) ? 0 : (ZProjection > LayersZCoordinate()[0]) ? ZBins - 1 : getZBinIndex(2, ZProjection) };
              int ZBinLow2{ (ZProjectionBin - mZSpan < 0) ? 0 : ZProjectionBin - mZSpan };
              int ZBinHigh2{ (ZProjectionBin + mZSpan > ZBins - 1) ? ZBins - 1 : ZProjectionBin + mZSpan };
              
              // int ZBinLow2  { 0 };
              // int ZBinHigh2  { ZBins - 1 };
              int PhiBinLow2{ (PhiBin1 - mPhiSpan < 0) ? PhiBins + PhiBin1 - mPhiSpan : PhiBin1 - mPhiSpan };       
              int PhiBinHigh2{ (PhiBin1 + mPhiSpan > PhiBins - 1) ? PhiBin1 + mPhiSpan - PhiBins : PhiBin1 + mPhiSpan };
              // int PhiBinLow2{ 0 };
              // int PhiBinHigh2{ PhiBins - 1 };

              clusters2 = selectClusters(mIndexTables[2], std::array<int, 4>{ ZBinLow2, PhiBinLow2, ZBinHigh2, PhiBinHigh2 });
              bool trackFound = false;

              for (int iRow2{ 0 }; iRow2 < clusters2.size(); ++iRow2) {
                for (int iCluster2{ std::get<0>(clusters2[iRow2]) }; iCluster2 < std::get<0>(clusters2[iRow2]) + std::get<1>(clusters2[iRow2]); ++iCluster2) {
                  float ZProjectionRefined { mClusters[0][iCluster0].zCoordinate +  (mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
                  if ( std::abs( mClusters[2][iCluster2].zCoordinate - ZProjectionRefined ) < mZCut &&
                       ( std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) > TwoPi - mPhiCut ) ) {

                    mTracklets.emplace_back(Line{
                      std::array<float, 3>{ mClusters[0][iCluster0].xCoordinate, mClusters[0][iCluster0].yCoordinate,
                                            mClusters[0][iCluster0].zCoordinate },
                      std::array<float, 3>{ mClusters[1][iCluster1].xCoordinate, mClusters[1][iCluster1].yCoordinate,
                                            mClusters[1][iCluster1].zCoordinate },
                                            mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID(),
                                            mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()});
                    trackFound = true;
                    mZDelta.push_back( ZProjection - mClusters[2][iCluster2].zCoordinate );
                    break;
                  }
                }
                if ( trackFound ) break;
              }
            }
          }
        }
      }
    }

    // end = std::chrono::system_clock::now();
    // int elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    // std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    // std::cout << "Finished computation at " << std::ctime(&end_time) << "elapsed time: " << elapsed_milliseconds
    //           << "ms\n";
    mTrackletsFound = true;
  }
}

void Vertexer::generateTracklets()
{ // brute force, for testing
  for ( auto cluster0 : mClusters[0] ) {
    for ( auto cluster1 : mClusters[1] ) {
      if ( (mEvent.getLayer(0).getClusterLabel(cluster0.clusterId).getTrackID() == mEvent.getLayer(1).getClusterLabel(cluster1.clusterId).getTrackID()) &&
           (std::abs(cluster0.phiCoordinate - cluster1.phiCoordinate) < mPhiCut || std::abs( cluster0.phiCoordinate - cluster1.phiCoordinate ) > TwoPi - mPhiCut)
         ) {
            for ( auto cluster2 : mClusters[2] ) {
              // float ZProjection{ mClusters[0][iCluster0].zCoordinate +  (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              // mZProjections.push_back(ZProjection);
              if ( mEvent.getLayer(0).getClusterLabel(cluster0.clusterId).getTrackID() == mEvent.getLayer(2).getClusterLabel(cluster2.clusterId).getTrackID() ) {
                mTracklets.emplace_back( Line{ std::array<float, 3> { cluster0.xCoordinate, cluster0.yCoordinate, cluster0.zCoordinate },
                  std::array<float, 3> { cluster1.xCoordinate, cluster1.yCoordinate, cluster1.zCoordinate }, std::array<float, 3> { cluster2.xCoordinate, cluster2.yCoordinate, cluster2.zCoordinate },
                  mEvent.getLayer(0).getClusterLabel(cluster0.clusterId).getTrackID(),
                  mEvent.getLayer(1).getClusterLabel(cluster1.clusterId).getTrackID()} );
                float ZProjection{ cluster0.zCoordinate + ( cluster2.rCoordinate - cluster0.rCoordinate ) * ( cluster1.zCoordinate - cluster0.zCoordinate ) / (cluster1.rCoordinate - cluster0.rCoordinate) };
                mZDelta.push_back( ZProjection - cluster2.zCoordinate );
                // mPhiProjections.push_back();
              }
            }
      }
    }
  }
  mTrackletsFound = true;
}

void Vertexer::findVertices()
{
  if (mTrackletsFound) {
    const int numTracklets{ static_cast<int>(mTracklets.size()) };
    mUsedTracklets.resize(numTracklets, false);

    for (int tracklet1{ 0 }; tracklet1 < numTracklets; ++tracklet1) {
      if (mUsedTracklets[tracklet1])
        continue;
      for (int tracklet2{ tracklet1 + 1 }; tracklet2 < numTracklets; ++tracklet2) {
        if (mUsedTracklets[tracklet2])
          continue;
        if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mPairCut) {
          mTrackletClusters.emplace_back(tracklet1, mTracklets[tracklet1], tracklet2, mTracklets[tracklet2]);
          std::array<float, 3> tmpVertex{ mTrackletClusters.back().getVertex() };
          if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
            mTrackletClusters.pop_back();
            break;
          }
          mUsedTracklets[tracklet1] = true;
          mUsedTracklets[tracklet2] = true;
          for (int tracklet3{ 0 }; tracklet3 < numTracklets; ++tracklet3) {
            if (mUsedTracklets[tracklet3])
              continue;
            if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mPairCut) {
              mTrackletClusters.back().add(tracklet3, mTracklets[tracklet3]);
              mUsedTracklets[tracklet3] = true;
              tmpVertex = mTrackletClusters.back().getVertex();
            }
          }
          break;
        }
      }
    }
    std::sort(mTrackletClusters.begin(), mTrackletClusters.end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    for (int iCluster1{ 0 }; iCluster1 < mTrackletClusters.size(); ++iCluster1) {
      std::array<float, 3> vertex1{ mTrackletClusters[iCluster1].getVertex() };
      std::array<float, 3> vertex2{};
      for (int iCluster2{ iCluster1 + 1 }; iCluster2 < mTrackletClusters.size(); ++iCluster2) {
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
            mTrackletClusters.erase(mTrackletClusters.begin() + iCluster2);
            --iCluster2;
          }
        }
      }
      for (int iCluster{ 0 }; iCluster < mTrackletClusters.size(); ++iCluster) {
        if (mTrackletClusters[iCluster].getSize() < mClusterContributorsCut && mTrackletClusters.size() > 1) {
          mTrackletClusters.erase(mTrackletClusters.begin() + iCluster);
          continue;
        }

        mVertices.emplace_back(mTrackletClusters[iCluster].getVertex());
      }
    }
  }
}

void Vertexer::printVertices()
{
  std::cout << "Number of found vertices: " << mVertices.size() << std::endl;
  for (auto& vertex : mVertices) {
    for (int i{ 0 }; i < 3; ++i) {
      std::cout << "coord: " << i << " -> " << vertex[i] << std::endl;
    }
  }
}

} // namespace CA
} // namespace ITS
} // namespace o2