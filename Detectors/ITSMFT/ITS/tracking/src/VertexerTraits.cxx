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

#include <iostream>

namespace o2
{
namespace ITS
{

using Constants::IndexTable::PhiBins;
using Constants::IndexTable::ZBins;
using Constants::ITS::LayersRCoordinate;
using Constants::ITS::LayersZCoordinate;
using Constants::Math::TwoPi;
using IndexTableUtils::getZBinIndex;

VertexerTraits::VertexerTraits() : mVrtParams{},
                                   mAverageClustersRadii{ std::array<float, 3>{ 0.f, 0.f, 0.f } },
                                   mMaxDirectorCosine3{ 0.f }
{
}

void VertexerTraits::reset()
{
  mTracklets.clear();
  mTrackletClusters.clear();
  mVertices.clear();
  mAverageClustersRadii = { 0.f, 0.f, 0.f };
  mMaxDirectorCosine3 = 0.f;
}

void VertexerTraits::initialise(const ROframe& event)
{
  reset();
  mEvent = const_cast<o2::ITS::ROframe*>(&event);
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumberVertexer; ++iLayer) {
    const auto& currentLayer{ event.getClustersOnLayer(iLayer) };
    const size_t clustersNum{ currentLayer.size() };
    if (clustersNum > 0) {
      mClusters[iLayer].clear();
      if (clustersNum > mClusters[iLayer].capacity()) {
        mClusters[iLayer].reserve(clustersNum);
      }
      for (auto iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        mClusters[iLayer].emplace_back(iLayer, currentLayer.at(iCluster));
        mAverageClustersRadii[iLayer] += mClusters[iLayer].back().rCoordinate;
      }
      mAverageClustersRadii[iLayer] *= 1.f / clustersNum;

      std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
        return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
      });
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
        mIndexTables[iLayer][iBin] = static_cast<int>(clustersNum);
      }
    }
    //else {
    //   mAverageClustersRadii[iLayer] = LayersRCoordinate()[iLayer];
    // }
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

void VertexerTraits::computeTracklets(const bool useMCLabel)
{
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
      std::array<int, 4>{ ZBinLow0, (PhiBin1 - mVrtParams.mPhiSpan < 0) ? PhiBins + (PhiBin1 - mVrtParams.mPhiSpan) : PhiBin1 - mVrtParams.mPhiSpan,
                          ZBinHigh0,
                          (PhiBin1 + mVrtParams.mPhiSpan > PhiBins) ? PhiBin1 + mVrtParams.mPhiSpan - PhiBins : PhiBin1 + mVrtParams.mPhiSpan });

    for (int iCluster1{ mIndexTables[1][iBin1] }; iCluster1 < mIndexTables[1][iBin1 + 1]; ++iCluster1) {
      bool trackFound = false;
      for (int iRow0{ 0 }; iRow0 < clusters0.size(); ++iRow0) {
        for (int iCluster0{ std::get<0>(clusters0[iRow0]) };
             iCluster0 < std::get<0>(clusters0[iRow0]) + std::get<1>(clusters0[iRow0]); ++iCluster0) {
          if (usedCluster0Flags[iCluster0])
            continue;
          if (std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) < mVrtParams.mPhiCut ||
              std::abs(mClusters[0][iCluster0].phiCoordinate - mClusters[1][iCluster1].phiCoordinate) >
                TwoPi - mVrtParams.mPhiCut) {
            float ZProjection{ mClusters[0][iCluster0].zCoordinate +
                               (mAverageClustersRadii[2] - mClusters[0][iCluster0].rCoordinate) *
                                 (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                 (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
            float ZProjectionInner{ mClusters[0][iCluster0].zCoordinate +
                                    (mClusters[0][iCluster0].rCoordinate) *
                                      (mClusters[1][iCluster1].zCoordinate - mClusters[0][iCluster0].zCoordinate) /
                                      (mClusters[1][iCluster1].rCoordinate - mClusters[0][iCluster0].rCoordinate) };
            if (std::abs(ZProjection) > (LayersZCoordinate()[0] + mVrtParams.mZCut))
              continue;
            int ZProjectionBin{ (ZProjection < -LayersZCoordinate()[0])
                                  ? 0
                                  : (ZProjection > LayersZCoordinate()[0]) ? ZBins - 1
                                                                           : getZBinIndex(2, ZProjection) };
            int ZBinLow2{ (ZProjectionBin - mVrtParams.mZSpan < 0) ? 0 : ZProjectionBin - mVrtParams.mZSpan };
            int ZBinHigh2{ (ZProjectionBin + mVrtParams.mZSpan > ZBins - 1) ? ZBins - 1 : ZProjectionBin + mVrtParams.mZSpan };
            // int ZBinLow2  { 0 };
            // int ZBinHigh2  { ZBins - 1 };
            int PhiBinLow2{ (PhiBin1 - mVrtParams.mPhiSpan < 0) ? PhiBins + PhiBin1 - mVrtParams.mPhiSpan : PhiBin1 - mVrtParams.mPhiSpan };
            int PhiBinHigh2{ (PhiBin1 + mVrtParams.mPhiSpan > PhiBins - 1) ? PhiBin1 + mVrtParams.mPhiSpan - PhiBins : PhiBin1 + mVrtParams.mPhiSpan };
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
                             (mEvent->getClusterLabels(0, mClusters[0][iCluster0]).getTrackID() ==
                                mEvent->getClusterLabels(2, mClusters[2][iCluster2]).getTrackID() &&
                              mEvent->getClusterLabels(0, mClusters[0][iCluster0]).getTrackID() ==
                                mEvent->getClusterLabels(1, mClusters[1][iCluster1]).getTrackID()) };
                float absDeltaPhi{ std::abs(mClusters[2][iCluster2].phiCoordinate -
                                            mClusters[1][iCluster1].phiCoordinate) };
                float absDeltaZ{ std::abs(mClusters[2][iCluster2].zCoordinate - ZProjectionRefined) };
                if (absDeltaZ < mVrtParams.mZCut && ((absDeltaPhi < mVrtParams.mPhiCut || std::abs(absDeltaPhi - TwoPi) < mVrtParams.mPhiCut) && testMC)) {
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
      if (Line::getDCA(mTracklets[tracklet1], mTracklets[tracklet2]) <= mVrtParams.mPairCut) {
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
          if (Line::getDistanceFromPoint(mTracklets[tracklet3], tmpVertex) < mVrtParams.mPairCut) {
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
      if (std::abs(vertex1[2] - vertex2[2]) < mVrtParams.mClusterCut) {

        float distance{ (vertex1[0] - vertex2[0]) * (vertex1[0] - vertex2[0]) +
                        (vertex1[1] - vertex2[1]) * (vertex1[1] - vertex2[1]) +
                        (vertex1[2] - vertex2[2]) * (vertex1[2] - vertex2[2]) };
        if (distance <= mVrtParams.mPairCut * mVrtParams.mPairCut) {
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
    if (mTrackletClusters[iCluster].getSize() < mVrtParams.mClusterContributorsCut && noClusters > 1) {
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
      mVertices.emplace_back(
        Point3D<float>{ mTrackletClusters[iCluster].getVertex()[0], mTrackletClusters[iCluster].getVertex()[1],
                        mTrackletClusters[iCluster].getVertex()[2] },
        mTrackletClusters[iCluster].getRMS2(),        // Symm matrix. Diagonal: RMS2 components,
                                                      // off-diagonal: square mean of projections on planes.
        mTrackletClusters[iCluster].getSize(),        // Contributors
        mTrackletClusters[iCluster].getAvgDistance2() // In place of chi2
        );
      mVertices.back().setTimeStamp(mEvent->getROFrameId());
      mEvent->addPrimaryVertex(mVertices.back().getX(), mVertices.back().getY(), mVertices.back().getZ());
    }
  }
}

void VertexerTraits::dumpVertexerTraits()
{
  std::cout << "Dump traits:" << std::endl;
  std::cout << "Tracklets found: " << mTracklets.size() << std::endl;
  std::cout << "Clusters of tracklets: " << mTrackletClusters.size() << std::endl;
  std::cout << "mVrtParams.mPairCut: " << mVrtParams.mPairCut << std::endl;
  std::cout << "Vertices found: " << mVertices.size() << std::endl;
}

} // namespace ITS
} // namespace o2
