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
/// \file VertexerTraits.cxx
/// \brief
///

#include "ITStracking/VertexerTraits.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include <iostream>

#define LAYER0_TO_LAYER1 0
#define LAYER1_TO_LAYER2 1

namespace o2
{
namespace its
{

using constants::IndexTable::PhiBins;
using constants::IndexTable::ZBins;
using constants::its::LayersRCoordinate;
using constants::its::LayersZCoordinate;
using constants::Math::TwoPi;
using IndexTableUtils::getZBinIndex;

void trackleterKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    // 0 2
  const std::vector<Cluster>& clustersCurrentLayer, // 1 1
  const std::array<int, ZBins * PhiBins + 1>& indexTableNext,
  const char layerOrder,
  const float phiCut,
  std::vector<Tracklet>& Tracklets,
  std::vector<int>& foundTracklets,
  const char isMc,
  const std::vector<int>& nextLayerMClabels,
  const std::vector<int>& currentLayerMClabels,
  const int maxTrackletsPerCluster = static_cast<int>(2e3))
{
  foundTracklets.resize(clustersCurrentLayer.size(), 0);
  // loop on layer1 clusters
  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int storedTracklets{ 0 };
    const Cluster currentCluster{ clustersCurrentLayer[iCurrentLayerClusterIndex] };
    const int layerIndex{ layerOrder == LAYER0_TO_LAYER1 ? 0 : 2 };
    const int4 selectedBinsRect{ VertexerTraits::getBinsRect2(currentCluster, layerIndex, 0.f, 50.f, phiCut) };
    if (selectedBinsRect.x != 0 || selectedBinsRect.y != 0 || selectedBinsRect.z != 0 || selectedBinsRect.w != 0) {
      int phiBinsNum{ selectedBinsRect.w - selectedBinsRect.y + 1 };
      if (phiBinsNum < 0) {
        phiBinsNum += PhiBins;
      }
      // loop on phi bins next layer
      for (int iPhiBin{ selectedBinsRect.y }, iPhiCount{ 0 }; iPhiCount < phiBinsNum; iPhiBin = ++iPhiBin == PhiBins ? 0 : iPhiBin, iPhiCount++) {
        const int firstBinIndex{ IndexTableUtils::getBinIndex(selectedBinsRect.x, iPhiBin) };
        const int firstRowClusterIndex{ indexTableNext[firstBinIndex] };
        const int maxRowClusterIndex{ indexTableNext[firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1] };
        // loop on clusters next layer
        for (int iNextLayerClusterIndex{ firstRowClusterIndex }; iNextLayerClusterIndex <= maxRowClusterIndex && iNextLayerClusterIndex < (int)clustersNextLayer.size(); ++iNextLayerClusterIndex) {
          const Cluster& nextCluster{ clustersNextLayer[iNextLayerClusterIndex] };
          const char testMC{ !isMc || (nextLayerMClabels[iNextLayerClusterIndex] == currentLayerMClabels[iCurrentLayerClusterIndex] && nextLayerMClabels[iNextLayerClusterIndex] != -1) };
          if (MATH_ABS(currentCluster.phiCoordinate - nextCluster.phiCoordinate) < phiCut && testMC) {
            if (storedTracklets < maxTrackletsPerCluster) {
              if (layerOrder == LAYER0_TO_LAYER1) {
                Tracklets.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, currentCluster);
              } else {
                Tracklets.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, currentCluster, nextCluster);
              }
              ++storedTracklets;
            }
          }
        }
      }
    }
    foundTracklets[iCurrentLayerClusterIndex] = storedTracklets;
  }
}

void trackletSelectionKernelSerial(
  const std::vector<Cluster>& clustersNextLayer,    //0
  const std::vector<Cluster>& clustersCurrentLayer, //1
  const std::vector<Cluster>& debugClustersLayer2,  //2
  const std::vector<Tracklet>& tracklets01,
  const std::vector<Tracklet>& tracklets12,
  const std::vector<int>& foundTracklets01,
  const std::vector<int>& foundTracklets12,
  std::vector<Line>& destTracklets,
  std::vector<std::array<float, 7>>& tlv,
  const float tanLambdaCut = 0.025f,
  const int maxTracklets = static_cast<int>(2e3))
{
  int offset01{ 0 };
  int offset12{ 0 };
  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < clustersCurrentLayer.size(); ++iCurrentLayerClusterIndex) {
    int validTracklets{ 0 };
    for (int iTracklet12{ offset12 }; iTracklet12 < offset12 + foundTracklets12[iCurrentLayerClusterIndex]; ++iTracklet12) {
      for (int iTracklet01{ offset01 }; iTracklet01 < offset01 + foundTracklets01[iCurrentLayerClusterIndex]; ++iTracklet01) {
        const float deltaTanLambda{ MATH_ABS(tracklets01[iTracklet01].tanLambda - tracklets12[iTracklet12].tanLambda) };
        if (deltaTanLambda < tanLambdaCut && validTracklets != maxTracklets) {
          assert(tracklets01[iTracklet01].secondClusterIndex == tracklets12[iTracklet12].firstClusterIndex);
          // tlv.push_back(std::array<float, 7>{ deltaTanLambda,
          //                                     clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].zCoordinate, clustersNextLayer[tracklets01[iTracklet01].firstClusterIndex].rCoordinate,
          //                                     clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].zCoordinate, clustersCurrentLayer[tracklets01[iTracklet01].secondClusterIndex].rCoordinate,
          //                                     debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].zCoordinate, debugClustersLayer2[tracklets12[iTracklet12].secondClusterIndex].rCoordinate });
          destTracklets.emplace_back(tracklets01[iTracklet01], clustersNextLayer.data(), clustersCurrentLayer.data());
          ++validTracklets;
        }
      }
    }
    offset01 += foundTracklets01[iCurrentLayerClusterIndex];
    offset12 += foundTracklets12[iCurrentLayerClusterIndex];
    // // if (validTracklets != maxTracklets) {
    // //   new (destTracklets + stride + validTracklets) Line(); // always complete line with empty one unless all spaces taken
    // // } else {
    // //   printf("[INFO]: Fulfilled all the space with tracklets.\n");
    // // }
  }
}

VertexerTraits::VertexerTraits() : mAverageClustersRadii{ std::array<float, 3>{ 0.f, 0.f, 0.f } },
                                   mMaxDirectorCosine3{ 0.f }
{
  // CUDA does not allow for dynamic initialization -> no constructor for VertexingParams
  mVrtParams.phiSpan = static_cast<int>(std::ceil(constants::IndexTable::PhiBins * mVrtParams.phiCut /
                                                  constants::Math::TwoPi));
  mVrtParams.zSpan = static_cast<int>(std::ceil(mVrtParams.zCut * constants::IndexTable::InverseZBinSize()[0]));
  setIsGPU(false);
}

void VertexerTraits::reset()
{
  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    mClusters[iLayer].clear();
    mIndexTables[iLayer].fill(0);
  }

  mTracklets.clear();
  mTrackletClusters.clear();
  mVertices.clear();
  mComb01.clear();
  mComb12.clear();
  mDeltaTanlambdas.clear();
  mCentroids.clear();
  mLinesData.clear();
  mAverageClustersRadii = { 0.f, 0.f, 0.f };
  mMaxDirectorCosine3 = 0.f;
}

std::vector<int> VertexerTraits::getMClabelsLayer(const int layer) const
{
  return mEvent->getTracksId(layer, mClusters[layer]);
}

void VertexerTraits::arrangeClusters(ROframe* event)
{
  mEvent = event;
  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumberVertexer; ++iLayer) {
    const auto& currentLayer{ event->getClustersOnLayer(iLayer) };
    const size_t clustersNum{ currentLayer.size() };
    if (clustersNum > 0) {
      if (clustersNum > mClusters[iLayer].capacity()) {
        mClusters[iLayer].reserve(clustersNum);
      }
      for (unsigned int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        mClusters[iLayer].emplace_back(iLayer, currentLayer.at(iCluster));
        mAverageClustersRadii[iLayer] += mClusters[iLayer].back().rCoordinate;
      }
      mAverageClustersRadii[iLayer] *= 1.f / clustersNum;

      std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
        return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
      });
      int previousBinIndex{ 0 };
      mIndexTables[iLayer][0] = 0;
      for (unsigned int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        const int currentBinIndex{ mClusters[iLayer][iCluster].indexTableBinIndex };
        if (currentBinIndex > previousBinIndex) {
          for (int iBin{ previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {
            mIndexTables[iLayer][iBin] = iCluster;
          }
          previousBinIndex = currentBinIndex;
        }
      }
      for (int iBin{ previousBinIndex + 1 }; iBin <= ZBins * PhiBins; iBin++) {
        mIndexTables[iLayer][iBin] = static_cast<int>(clustersNum);
      }
    }
  }
  mDeltaRadii10 = mAverageClustersRadii[1] - mAverageClustersRadii[0];
  mDeltaRadii21 = mAverageClustersRadii[2] - mAverageClustersRadii[1];
  mMaxDirectorCosine3 =
    LayersZCoordinate()[2] / std::sqrt(LayersZCoordinate()[2] * LayersZCoordinate()[2] +
                                       (mDeltaRadii10 + mDeltaRadii21) * (mDeltaRadii10 + mDeltaRadii21));
}

const std::vector<std::pair<int, int>> VertexerTraits::selectClusters(const std::array<int, ZBins * PhiBins + 1>& indexTable,
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

/* void VertexerTraits::computeTracklets(const bool useMCLabel)
{
  if (useMCLabel)
    std::cout << "Running in MOntecarlo check mode\n";
  std::vector<std::pair<int, int>> clusters0, clusters2;
  // std::vector<bool> mUsedClusters[1], mUsedClusters[0];
  mUsedClusters[1].resize(mClusters[2].size(), false);
  mUsedClusters[0].resize(mClusters[0].size(), false);

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
      std::array<int, 4>{ ZBinLow0, (PhiBin1 - mVrtParams.phiSpan < 0) ? PhiBins + (PhiBin1 - mVrtParams.phiSpan) : PhiBin1 - mVrtParams.phiSpan,
                          ZBinHigh0,
                          (PhiBin1 + mVrtParams.phiSpan > PhiBins) ? PhiBin1 + mVrtParams.phiSpan - PhiBins : PhiBin1 + mVrtParams.phiSpan });

    for (int iCluster1{ mIndexTables[1][iBin1] }; iCluster1 < mIndexTables[1][iBin1 + 1]; ++iCluster1) {
      bool trackFound = false;
      for (int iRow0{ 0 }; iRow0 < clusters0.size(); ++iRow0) {
        for (int iCluster0{ std::get<0>(clusters0[iRow0]) };
             iCluster0 < std::get<0>(clusters0[iRow0]) + std::get<1>(clusters0[iRow0]); ++iCluster0) {
          if (mUsedClusters[0][iCluster0])
            continue;
          if (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mVrtParams.phiCut ||
              std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) >
                TwoPi - mVrtParams.phiCut) {
            float ZProjection{ mClusters[0][iCluster0].zCoordinate +
                               (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) *
                                 (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                 (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
            float ZProindexTableNextectionInner{ mClusters[0][iCluster0].zCoordinate +
                      indexTableNext             (mClusters[0][iCluster0].rCoordinate) *
                                      (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                      (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
            if (std::abs(ZProjection) > (LayersZCoordinate()[0] + mVrtParams.zCut))
              continue;
            int ZProjectionBin{ (ZProjection < -LayersZCoordinate()[0])
                                  ? 0
                                  : (ZProjection > LayersZCoordinate()[0]) ? ZBins - 1
                                                                           : getZBinIndex(2, ZProjection) };
            int ZBinLow2{ (ZProjectionBin - mVrtParams.zSpan < 0) ? 0 : ZProjectionBin - mVrtParams.zSpan };
            int ZBinHigh2{ (ZProjectionBin + mVrtParams.zSpan > ZBins - 1) ? ZBins - 1 : ZProjectionBin + mVrtParams.zSpan };
            // int ZBinLow2  { 0 };
            // int ZBinHigh2  { ZBins - 1 };
            int PhiBinLow2{ (PhiBin1 - mVrtParams.phiSpan < 0) ? PhiBins + PhiBin1 - mVrtParams.phiSpan : PhiBin1 - mVrtParams.phiSpan };
            int PhiBinHigh2{ (PhiBin1 + mVrtParams.phiSpan > PhiBins - 1) ? PhiBin1 + mVrtParams.phiSpan - PhiBins : PhiBin1 + mVrtParams.phiSpan };
            // int PhiBinLow2{ 0 };
            // int PhiBinHigh2{ PhiBins - 1 };
            clusters2 =
              selectClusters(mIndexTables[2], std::array<int, 4>{ ZBinLow2, PhiBinLow2, ZBinHigh2, PhiBinHigh2 });
            for (int iRow2{ 0 }; iRow2 < clusters2.size(); ++iRow2) {
              for (int iCluster2{ std::get<0>(clusters2[iRow2]) };
                   iCluster2 < std::get<0>(clusters2[iRow2]) + std::get<1>(clusters2[iRow2]); ++iCluster2) {
                if (mUsedClusters[1][iCluster2])
                  continue;
                float ZProjectionRefined{
                  mClusters[0][iCluster0].zCoordinate +
                  (mClusters[2][iCluster2].rCoordinate - mClusters[0][iCluster0].rCoordinate) *
                    (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                    (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate)
                };
                bool testMC{ !useMCLabel ||
                             (mEvent->getClusterLabels(0, mClusters[0][iCluster0]).getTrackID() ==
                                mEvent->getClusterLabels(2, mClusters[2][iCluster2]).getTrackID() &&
                              mEvent->getClusterLabels(0, mClusters[0][iCluster0]).getTrackID() ==
                                mEvent->getClusterLabels(1, mClusters[1][iCluster1]).getTrackID()) };
                float absDeltaPhi{ std::abs(mClusters[2][iCluster2].phiCoordinate -
                                            mClusters[1][iCluster1].phiCoordinate) };
                float absDeltaZ{ std::abs(mClusters[2][iCluster2].zCoordinate - ZProjectionRefined) };
                if (absDeltaZ < mVrtParams.zCut && ((absDeltaPhi < mVrtParams.phiCut || std::abs(absDeltaPhi - TwoPi) < mVrtParams.phiCut) && testMC)) {
                  mTracklets.emplace_back(Line{
                    std::array<float, 3>{ mClusters[0][iCluster0].xCoordinate, mClusters[0][iCluster0].yCoordinate,
                                          mClusters[0][iCluster0].zCoordinate },
                    std::array<float, 3>{ mClusters[1][iCluster1].xCoordinate, mClusters[1][iCluster1].yCoordinate,
                                          mClusters[1][iCluster1].zCoordinate } });
                  if (std::abs(mTracklets.back().cosinesDirector[2]) < mMaxDirectorCosine3) {
                    mUsedClusters[0][iCluster0] = true;
                    mUsedClusters[1][iCluster2] = true;
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
}*/

void VertexerTraits::computeTrackletsPureMontecarlo()
{

  std::cout << "Running in Montecarlo trivial mode\n";
  std::cout << "clusters on L0: " << mClusters[0].size() << " clusters on L1: " << mClusters[1].size() << " clusters on L2: " << mClusters[2].size() << std::endl;

  std::vector<int> foundTracklets01;
  std::vector<int> foundTracklets12;

  std::vector<int> labelsMC0 = getMClabelsLayer(0);
  std::vector<int> labelsMC1 = getMClabelsLayer(1);
  std::vector<int> labelsMC2 = getMClabelsLayer(2);

  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < mClusters[0].size(); ++iCurrentLayerClusterIndex) {
    auto& cluster{ mClusters[0][iCurrentLayerClusterIndex] };
    //const int2 selectedBinsRect{ VertexerTraits::getPhiBins(1, cluster.phiCoordinate, mVrtParams.phiCut) };
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{ mClusters[1][iNextLayerClusterIndex] };
      if (/*MATH_ABS(cluster.phiCoordinate - nextCluster.phiCoordinate) < mVrtParams.phiCut &&*/ labelsMC1[iNextLayerClusterIndex] == labelsMC0[iCurrentLayerClusterIndex] && labelsMC1[iNextLayerClusterIndex] != -1
          /*mEvent->getClusterLabels(0, cluster).getTrackID() == mEvent->getClusterLabels(1, nextCluster).getTrackID() && mEvent->getClusterLabels(0, cluster).getTrackID() != -1*/)
        mComb01.emplace_back(iCurrentLayerClusterIndex, iNextLayerClusterIndex, cluster, nextCluster);
    }
  }

  for (unsigned int iCurrentLayerClusterIndex{ 0 }; iCurrentLayerClusterIndex < mClusters[2].size(); ++iCurrentLayerClusterIndex) {
    auto& cluster{ mClusters[2][iCurrentLayerClusterIndex] };
    //const int2 selectedBinsRect{ VertexerTraits::getPhiBins(1, cluster.phiCoordinate, mVrtParams.phiCut) };
    for (unsigned int iNextLayerClusterIndex = 0; iNextLayerClusterIndex < mClusters[1].size(); iNextLayerClusterIndex++) {
      const Cluster& nextCluster{ mClusters[1][iNextLayerClusterIndex] };
      if (/*MATH_ABS(cluster.phiCoordinate - nextCluster.phiCoordinate) < mVrtParams.phiCut && (*/ labelsMC1[iNextLayerClusterIndex] == labelsMC2[iCurrentLayerClusterIndex] && labelsMC1[iNextLayerClusterIndex] != -1
          /*mEvent->getClusterLabels(2, cluster).getTrackID() == mEvent->getClusterLabels(1, nextCluster).getTrackID() && mEvent->getClusterLabels(2, cluster).getTrackID() != -1*/)
        mComb12.emplace_back(iNextLayerClusterIndex, iCurrentLayerClusterIndex, nextCluster, cluster);
    }
  }

  for (auto& trklet01 : mComb01) {
    for (auto& trklet12 : mComb12) {
      if (trklet01.secondClusterIndex == trklet12.firstClusterIndex) {
        const float deltaTanLambda{ MATH_ABS(trklet01.tanLambda - trklet12.tanLambda) };
        mDeltaTanlambdas.push_back(std::array<float, 7>{ deltaTanLambda,
                                                         mClusters[0][trklet01.firstClusterIndex].zCoordinate, mClusters[0][trklet01.firstClusterIndex].rCoordinate,
                                                         mClusters[1][trklet01.secondClusterIndex].zCoordinate, mClusters[1][trklet01.secondClusterIndex].rCoordinate,
                                                         mClusters[2][trklet12.secondClusterIndex].zCoordinate, mClusters[2][trklet12.secondClusterIndex].rCoordinate });
      }
    }
  }

  for (auto& trk : mComb01) {
    mTracklets.emplace_back(trk, mClusters[0].data(), mClusters[1].data());
  }
}

void VertexerTraits::computeTracklets(const bool useMCLabel)
{
  // computeTrackletsPureMontecarlo();
  if (useMCLabel)
    std::cout << "Running in Montecarlo check mode\n";
  std::cout << "clusters on L0: " << mClusters[0].size() << " clusters on L1: " << mClusters[1].size() << " clusters on L2: " << mClusters[2].size() << std::endl;

  std::vector<int> foundTracklets01;
  std::vector<int> foundTracklets12;

  ///TODO: Ugly hack!! The labels should be optionals in the trackleter kernel
  std::vector<int> labelsMC0 = useMCLabel ? getMClabelsLayer(0) : std::vector<int>();
  std::vector<int> labelsMC1 = useMCLabel ? getMClabelsLayer(1) : std::vector<int>();
  std::vector<int> labelsMC2 = useMCLabel ? getMClabelsLayer(2) : std::vector<int>();

  trackleterKernelSerial(
    mClusters[0],
    mClusters[1],
    mIndexTables[0],
    LAYER0_TO_LAYER1,
    mVrtParams.phiCut,
    mComb01,
    foundTracklets01,
    useMCLabel,
    labelsMC0,
    labelsMC1);

  trackleterKernelSerial(
    mClusters[2],
    mClusters[1],
    mIndexTables[2],
    LAYER1_TO_LAYER2,
    mVrtParams.phiCut,
    mComb12,
    foundTracklets12,
    useMCLabel,
    labelsMC2,
    labelsMC1);

  trackletSelectionKernelSerial(
    mClusters[0],
    mClusters[1],
    mClusters[2],
    mComb01,
    mComb12,
    foundTracklets01,
    foundTracklets12,
    mTracklets,
    mDeltaTanlambdas);
}

void VertexerTraits::computeVertices()
{
  const int numTracklets{ static_cast<int>(mTracklets.size()) };
  std::vector<bool> usedTracklets{};
  usedTracklets.resize(mTracklets.size(), false);
  for (int tracklet1{ 0 }; tracklet1 < numTracklets; ++tracklet1) {
    if (usedTracklets[tracklet1])
      continue;
    for (int tracklet2{ tracklet1 + 1 }; tracklet2 < numTracklets; ++tracklet2) {
      if (usedTracklets[tracklet2])
        continue;
      if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mVrtParams.pairCut) {
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
          if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mVrtParams.pairCut) {
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
      if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.clusterCut) {

        float distance{ (vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                        (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                        (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]) };
        if (distance <= mVrtParams.pairCut * mVrtParams.pairCut) {
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
    if (mTrackletClusters[iCluster].getSize() < mVrtParams.clusterContributorsCut && noClusters > 1) {
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
      mVertices.emplace_back(mTrackletClusters[iCluster].getVertex()[0],
                             mTrackletClusters[iCluster].getVertex()[1],
                             mTrackletClusters[iCluster].getVertex()[2],
                             mTrackletClusters[iCluster].getRMS2(),         // Symm matrix. Diagonal: RMS2 components,
                                                                            // off-diagonal: square mean of projections on planes.
                             mTrackletClusters[iCluster].getSize(),         // Contributors
                             mTrackletClusters[iCluster].getAvgDistance2(), // In place of chi2
                             mEvent->getROFrameId());
      mEvent->addPrimaryVertex(mVertices.back().mX, mVertices.back().mY, mVertices.back().mZ);
    }
  }
}

void VertexerTraits::dumpVertexerTraits()
{
  std::cout << "Dump traits:" << std::endl;
  std::cout << "Tracklets found: " << mTracklets.size() << std::endl;
  std::cout << "Clusters of tracklets: " << mTrackletClusters.size() << std::endl;
  std::cout << "mVrtParams.pairCut: " << mVrtParams.pairCut << std::endl;
  std::cout << "Vertices found: " << mVertices.size() << std::endl;
}

VertexerTraits* createVertexerTraits()
{
  return new VertexerTraits;
}

void VertexerTraits::processLines()
{
  for (unsigned int iLine1{ 0 }; iLine1 < mTracklets.size(); ++iLine1) {
    auto line1 = mTracklets[iLine1];
    for (unsigned int iLine2{ iLine1 + 1 }; iLine2 < mTracklets.size(); ++iLine2) {
      auto line2 = mTracklets[iLine2];
      ClusterLines cluster{ -1, line1, -1, line2 };
      auto vtx = cluster.getVertex();
      if (vtx[0] * vtx[0] + vtx[1] * vtx[1] < 1.98 * 1.98) {
        mCentroids.push_back(std::array<float, 4>{ vtx[0], vtx[1], vtx[2], Line::getDCA(line1, line2) });
      }
    }
    mLinesData.push_back(Line::getDCAComponents(line1, std::array<float, 3>{ 0., 0., 0. }));
  }
}

} // namespace its
} // namespace o2
