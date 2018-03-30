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
    if (mClusters[iLayer].size() != 0) {
      const float inverseNumberOfClusters { 1.f/mClusters[iLayer].size() };
      for ( auto& cluster : mClusters[iLayer] ) {
        mAverageClustersRadii[iLayer] += cluster.rCoordinate;
      }
      mAverageClustersRadii[iLayer] *= inverseNumberOfClusters;
    } else {
      mAverageClustersRadii[iLayer] = LayersRCoordinate()[iLayer];
    }
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

void Vertexer::findTracklets( const bool useMCLabel )
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
        // std::cout<<"-> "<<iCluster1<<"-> "<<mClusters[1][iCluster1].clusterId<<"-> "<<mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()<<std::endl;
        bool trackFound = false;
        for (int iRow0{ 0 }; iRow0 < clusters0.size(); ++iRow0) {
          for (int iCluster0{ std::get<0>(clusters0[iRow0]) }; iCluster0 < std::get<0>(clusters0[iRow0]) + std::get<1>(clusters0[iRow0]); ++iCluster0) {

            if (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) > TwoPi - mPhiCut) {

              float ZProjection{ mClusters[0][iCluster0].zCoordinate +  (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              float ZProjectionInner{ mClusters[0][iCluster0].zCoordinate +  (mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };

              if (std::abs(ZProjection) > (LayersZCoordinate()[0] + mZCut))
                continue;
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
              

              for (int iRow2{ 0 }; iRow2 < clusters2.size(); ++iRow2) {
                for (int iCluster2{ std::get<0>(clusters2[iRow2]) }; iCluster2 < std::get<0>(clusters2[iRow2]) + std::get<1>(clusters2[iRow2]); ++iCluster2) {

                  float ZProjectionRefined { mClusters[0][iCluster0].zCoordinate +  (mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
                  bool testMC { !useMCLabel || ( mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() == mEvent.getLayer(2).getClusterLabel(mClusters[2][iCluster2].clusterId).getTrackID() && 
                             mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() == mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID() ) 
                  };
                  if ( std::abs( mClusters[2][iCluster2].zCoordinate - ZProjectionRefined ) < mZCut &&
                       (std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || 
                        std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) > TwoPi - mPhiCut) &&
                        testMC
                     ) 
                  {
                    mTracklets.emplace_back(Line{
                                                 std::array<float, 3>{ mClusters[0][iCluster0].xCoordinate, mClusters[0][iCluster0].yCoordinate,
                                                   mClusters[0][iCluster0].zCoordinate },
                                                 std::array<float, 3>{ mClusters[1][iCluster1].xCoordinate, mClusters[1][iCluster1].yCoordinate,
                                                   mClusters[1][iCluster1].zCoordinate },
                                                 mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID(),
                                                 mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()
                                                }
                                            );
                    // std::cout<<"->"<<mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID();
                    trackFound = true;
                    // mZDelta.push_back( ZProjection - mClusters[2][iCluster2].zCoordinate ); 
                    break;
                  }
                }
                if ( trackFound )
                  break;
              }
            }
            if ( trackFound )
              break;
          }
          if ( trackFound )
            break;
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
  int prevID = -99;
  for ( int iCluster0 { 0 }; iCluster0 < mClusters[0].size(); ++iCluster0 ) {
    for ( int iCluster1 { 0 }; iCluster1 < mClusters[1].size(); ++iCluster1 ) {
      if ( (mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() == mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()) &&
           (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || std::abs( mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate ) > TwoPi - mPhiCut)
         ) 
      {
        bool trackFound { false }; 
        for ( int iCluster2 { 0 }; iCluster2 < mClusters[2].size(); ++iCluster2 ) {
          float ZProjectionRefined { mClusters[0][iCluster0].zCoordinate +  (mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate) * (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
          if ( std::abs( mClusters[2][iCluster2].zCoordinate - ZProjectionRefined ) < mZCut &&
             ( std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mPhiCut || std::abs(mClusters[2][iCluster2].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) > TwoPi - mPhiCut ) &&
             ( mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID() == mEvent.getLayer(2).getClusterLabel(mClusters[2][iCluster2].clusterId).getTrackID())) {
            trackFound = true;
            mTracklets.emplace_back( Line{ std::array<float, 3> { mClusters[0][iCluster0].xCoordinate, mClusters[0][iCluster0].yCoordinate, mClusters[0][iCluster0].zCoordinate },
              std::array<float, 3> { mClusters[1][iCluster1].xCoordinate, mClusters[1][iCluster1].yCoordinate, mClusters[1][iCluster1].zCoordinate }, std::array<float, 3> { mClusters[2][iCluster2].xCoordinate, mClusters[2][iCluster2].yCoordinate, mClusters[2][iCluster2].zCoordinate },
                mEvent.getLayer(0).getClusterLabel(mClusters[0][iCluster0].clusterId).getTrackID(), mEvent.getLayer(1).getClusterLabel(mClusters[1][iCluster1].clusterId).getTrackID()} );
              float ZProjection{ mClusters[0][iCluster0].zCoordinate + ( mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate ) * ( mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate ) / (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
              mZDelta.push_back( ZProjection - mClusters[2][iCluster2].zCoordinate );
              break;
          }
        }
        if ( trackFound )
          break; 
      }
    }
  }
  mTrackletsFound = true;
}

void Vertexer::dumpTracklets() {
  /* for ( auto& cluster : mClusters[0]) {
    if ( mEvent.getLayer(0).getClusterLabel(cluster.clusterId).getTrackID() == 60 ) {
      std::cout<<"x: "<<cluster.xCoordinate<<" y: "<<cluster.yCoordinate<<" z: "<<cluster.zCoordinate<<std::endl;
    }
  }*/
  for ( auto& tracklet : mTracklets ) {
    std::cout<<"or id: "<< tracklet.originID << "\t de id: " << tracklet.destinID <<std::endl;
  }
}

void Vertexer::findVertices()
{
  if (mTrackletsFound) {
    const int numTracklets{ static_cast<int>(mTracklets.size()) };
    mUsedTracklets.resize(mTracklets.size(), false);
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
              // tmpVertex = mTrackletClusters.back().getVertex();
              //std::array<float, 3> tmpVertex0 = mTrackletClusters.back().getVertex();            
              mTrackletClusters.back().add(tracklet3, mTracklets[tracklet3]);
              mUsedTracklets[tracklet3] = true;
              tmpVertex = mTrackletClusters.back().getVertex();
              // if (tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] > 4.f) {
                /*std::cout<<"Dump first tracklet:\n\torigin: "<<mTracklets[tracklet1].originID<<"\n\t\tx: "<<mTracklets[tracklet1].originPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet1].originPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet1].originPoint[2]<<"\n\tdestination: "<<mTracklets[tracklet1].destinID<<"\n\t\tx: "<<mTracklets[tracklet1].destinationPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet1].destinationPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet1].destinationPoint[2]<<std::endl;
                std::cout<<"\tcosines:\n\t\tcd1: "<<mTracklets[tracklet1].cosinesDirector[0]<<"\n\t\tcd2: "<<mTracklets[tracklet1].cosinesDirector[1]<<"\n\t\tcd3: "<<mTracklets[tracklet1].cosinesDirector[2]<<std::endl;
                std::cout<<"\ttracklet id: "<<tracklet1<<std::endl;

                std::cout<<"Dump second tracklet:\n\torigin: "<<mTracklets[tracklet2].originID<<"\n\t\tx: "<<mTracklets[tracklet2].originPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet2].originPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet2].originPoint[2]<<"\n\tdestination: "<<mTracklets[tracklet2].destinID<<"\n\t\tx: "<<mTracklets[tracklet2].destinationPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet2].destinationPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet2].destinationPoint[2]<<std::endl;
                std::cout<<"\tcosines:\n\t\tcd1: "<<mTracklets[tracklet2].cosinesDirector[0]<<"\n\t\tcd2: "<<mTracklets[tracklet2].cosinesDirector[1]<<"\n\t\tcd3: "<<mTracklets[tracklet2].cosinesDirector[2]<<std::endl;
                std::cout<<"\ttracklet id: "<<tracklet2<<std::endl;

                
                std::cout<<"Dump third tracklet:\n\torigin: "<<mTracklets[tracklet3].originID<<"\n\t\tx: "<<mTracklets[tracklet3].originPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet3].originPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet3].originPoint[2]<<"\n\tdestination: "<<mTracklets[tracklet3].destinID<<"\n\t\tx: "<<mTracklets[tracklet3].destinationPoint[0]<<"\n\t\ty: "<<mTracklets[tracklet3].destinationPoint[1]<<"\n\t\tz: "<<mTracklets[tracklet3].destinationPoint[2]<<std::endl;
                std::cout<<"\tcosines:\n\t\tcd1: "<<mTracklets[tracklet3].cosinesDirector[0]<<"\n\t\tcd2: "<<mTracklets[tracklet3].cosinesDirector[1]<<"\n\t\tcd3: "<<mTracklets[tracklet3].cosinesDirector[2]<<std::endl;
                std::cout<<"\ttracklet id: "<<tracklet3<<std::endl;

                std::cout << "-- radius: " << tmpVertex0[0] * tmpVertex0[0] + tmpVertex0[1] * tmpVertex0[1] << "\tradius after add: " << tmpVertex[0] * tmpVertex[0] + tmpVertex[1] * tmpVertex[1] << "\tdistance third tracklet from vtx: " << Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) << std::endl;
                */
                // break;
              // }
            }
          }
          break;
        }
      }
    }
    std::cout<<"clusters: "<<mTrackletClusters.size()<<std::endl;
    std::sort(mTrackletClusters.begin(), mTrackletClusters.end(),
              [](ClusterLines& cluster1, ClusterLines& cluster2) { return cluster1.getSize() > cluster2.getSize(); });
    // std::cout<<"ClusterSize: "<<mTrackletClusters.size()<<std::endl;
    int noClusters { static_cast<int>( mTrackletClusters.size() ) };
    for (int iCluster1{ 0 }; iCluster1 < noClusters; ++iCluster1) {
      // std::cout<<"\t Size of cluster: "<<iCluster1<<" is: "<<mTrackletClusters[iCluster1].getSize()<<std::endl;
      std::array<float, 3> vertex1{ mTrackletClusters[iCluster1].getVertex() };
      std::array<float, 3> vertex2{};
      // std::cout<<"v1 >> x: "<<vertex1[0]<<" y: "<<vertex1[1]<<" z: "<<vertex1[2]<<std::endl;
      for (int iCluster2{ iCluster1 + 1 }; iCluster2 < noClusters; ++iCluster2) {
        vertex2 = mTrackletClusters[iCluster2].getVertex();
        if (std::abs(vertex1[2] - vertex2[2]) < mClusterCut) {
          
          float distance{ (vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                          (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                          (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]) };
          std::cout<<"Z distance between vertices: "<<std::abs(vertex1[2] - vertex2[2])<<" Squared distance between vertices: "<<distance<<std::endl;
          // std::cout<<"\tv2 >> x: "<<vertex2[0]<<" y: "<<vertex2[1]<<" z: "<<vertex2[2]<<std::endl;
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
    // std::cout<<"noClusters "<<noClusters<<std::endl;
    for (int iCluster{ 0 }; iCluster < noClusters; ++iCluster) {
      std::cout<<"\tnoClusters: "<<noClusters<<"\tsize: "<<mTrackletClusters.size()<<std::endl;
      if ( mTrackletClusters[iCluster].getSize() < mClusterContributorsCut && noClusters > 1 ) {
        std::cout<<"===> "<<mTrackletClusters[iCluster].getSize()<<" < "<<mClusterContributorsCut<<std::endl;
        mTrackletClusters.erase(mTrackletClusters.begin() + iCluster);
        // std::cout<<"Erased, size is: "<<mTrackletClusters.size()<<std::endl;
        noClusters--;
        continue;
      }
      // std::cout<<"Emplacing vertex of cluster: "<<iCluster<<std::endl;
      float dist { 0. };
      // debug
      // std::cout<<"number of lines: "<<mTrackletClusters[iCluster].getSize()<<std::endl;
      for ( auto& line : mTrackletClusters[iCluster].mLines ) {
        dist += Line::getDistanceFromPoint(line, mTrackletClusters[iCluster].getVertex()) / mTrackletClusters[iCluster].getSize();
        // if ( 1 ) {
        std::cout<<"-> "<<iCluster<<"\n\tox: "<<line.originPoint[0]<<"\t"<<"oy: "<<line.originPoint[1]<<"\t"<<"oz: "<<line.originPoint[2]<<"\n\tdx: "<<line.destinationPoint[0]<<"\t"<<"dy: "<<line.destinationPoint[1]<<"\t"<<"dz: "<<line.destinationPoint[2]
          <<"\n\tcd1: "<<line.cosinesDirector[0]<<"\n\tcd2: "<<line.cosinesDirector[1]<<"\n\tcd3: "<<line.cosinesDirector[2]<<std::endl;
        // }
      }
      //
      if (mTrackletClusters[iCluster].getVertex()[0] * mTrackletClusters[iCluster].getVertex()[0] + mTrackletClusters[iCluster].getVertex()[1] * mTrackletClusters[iCluster].getVertex()[1] < 1.98*1.98 )
        mVertices.emplace_back(std::make_tuple(mTrackletClusters[iCluster].getVertex(), mTrackletClusters[iCluster].getSize(), dist));
    }
  }
}

} // namespace CA
} // namespace ITS
} // namespace o2

/*

// Try to find all the primary vertices in the current 
  fNoVertices=0;
  FindTracklets();
  if(fNoLines<2) { 
    //fVertices.push_back(AliESDVertex());
    return;// AliESDVertex();
  }
  
  //  fVertices.push_back(AliVertexerTracks::TrackletVertexFinder(&fLines,1));
  //fNoVertices=1;
  fUsedLines=new Short_t[fNoLines];
  for(UInt_t i=0;i<fNoLines;++i) fUsedLines[i]=-1;
  
  fNoClusters=0;
  for(UInt_t i1=0;i1<fNoLines;++i1) {
    if(fUsedLines[i1]!=-1) continue;
    AliStrLine* line1 = (AliStrLine*)fLines.At(i1);
    for(UInt_t i2=i1+1;i2<fNoLines;++i2) {
      if(fUsedLines[i2]!=-1) continue;
      AliStrLine* line2 = (AliStrLine*)fLines.At(i2);
      if(line1->GetDCA(line2)<=fPairCut) {
	      //cout << fNoClusters <<" " << i1 << " " << i2 << " ";
	      new(fLinesClusters[fNoClusters])AliITSUClusterLines(i1,line1,i2,line2);
	      AliITSUClusterLines* current=(AliITSUClusterLines*)fLinesClusters.At(fNoClusters);
	      Double_t p[3];
	      current->GetVertex(p);
	      if((p[0]*p[0]+p[1]*p[1])>=4) { // Beam pipe check
	        fLinesClusters.RemoveAt(fNoClusters);
	        fLinesClusters.Compress();
	        break; // i2 for loop
	      }
	      fUsedLines[i1]=fNoClusters;
	      fUsedLines[i2]=fNoClusters;
	      for(UInt_t i3=0;i3<fNoLines;++i3) {
	        if(fUsedLines[i3]!=-1) continue;
	        AliStrLine *line3 = (AliStrLine*)fLines.At(i3);
	        //cout << p[0] << " " << p[1] << " " << p[2] << endl;
	        //line3->PrintStatus();
	        if(line3->GetDistFromPoint(p)<=fPairCut) {
	          //cout << i3 << " ";
	          current->Add(i3,line3);
	          fUsedLines[i3]=fNoClusters;
	          current->GetVertex(p);
	        }
        }
	      ++fNoClusters;
	      //cout << endl;
	      break;
      }
    } // i2
  } // i1
  
  fLinesClusters.Sort();

  for(UInt_t i0=0;i0<fNoClusters; ++i0) {
    Double_t p0[3],p1[3];
    AliITSUClusterLines *clu0 = (AliITSUClusterLines*)fLinesClusters.At(i0);
    clu0->GetVertex(p0);
    for(UInt_t i1=i0+1;i1<fNoClusters; ++i1) {
      AliITSUClusterLines *clu1 = (AliITSUClusterLines*)fLinesClusters.At(i1);
      clu1->GetVertex(p1);
      if (TMath::Abs(p0[2]-p1[2])<=fClusterCut) {
	      Double_t distance=(p0[0]-p1[0])*(p0[0]-p1[0])+(p0[1]-p1[1])*(p0[1]-p1[1])+(p0[2]-p1[2])*(p0[2]-p1[2]);
	      //Bool_t flag=kFALSE;
	      if(distance<=fPairCut*fPairCut) {
	        UInt_t n=0;
	        Int_t *labels=clu1->GetLabels(n);
	        for(UInt_t icl=0; icl<n; ++icl) clu0->Add(labels[icl],(AliStrLine*)fLines.At(labels[icl]));
	          clu0->GetVertex(p0);
	          //flag=kTRUE;
	      }
	      fLinesClusters.RemoveAt(i1);
	      fLinesClusters.Compress();
	      fNoClusters--;
	      i1--;
	      //if(flag) i1=10;
      }
    }
  }

  fVertices=new AliESDVertex[fNoClusters];
  for(UInt_t i0=0; i0<fNoClusters; ++i0) {
    AliITSUClusterLines *clu0 = (AliITSUClusterLines*)fLinesClusters.At(i0);
    Int_t size=clu0->GetSize();
    if(size<fClusterContribCut&&fNoClusters>1) {
      fLinesClusters.RemoveAt(i0);
      fLinesClusters.Compress();
      fNoClusters--;
      continue;
    } 
    Double_t p0[3],cov[6];
    clu0->GetVertex(p0);
    clu0->GetCovMatrix(cov);
    if((p0[0]*p0[0]+p0[1]*p0[1])<1.98*1.98) {
      fVertices[fNoVertices++]=AliESDVertex(p0,cov,99999.,size);   
    }
  }
  
return;// AliVertexerTracks::TrackletVertexFinder(&fLines,0);
*/