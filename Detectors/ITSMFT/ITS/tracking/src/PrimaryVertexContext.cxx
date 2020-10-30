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
/// \file PrimaryVertexContext.cxx
/// \brief
///

#include "ITStracking/PrimaryVertexContext.h"

#include <iostream>

namespace o2
{
namespace its
{

void PrimaryVertexContext::initialise(const MemoryParameters& memParam, const TrackingParameters& trkParam,
                                      const std::vector<std::vector<Cluster>>& cl, const std::array<float, 3>& pVtx, const int iteration)
{

  struct ClusterHelper {
    float phi;
    float r;
    int bin;
    int ind;
  };

  mPrimaryVertex = {pVtx[0], pVtx[1], pVtx[2]};

  if (iteration == 0) {

    std::vector<ClusterHelper> cHelper;

    mMinR.resize(trkParam.NLayers, 10000.);
    mMaxR.resize(trkParam.NLayers, -1.);
    mClusters.resize(trkParam.NLayers);
    mUsedClusters.resize(trkParam.NLayers);
    mCells.resize(trkParam.CellsPerRoad());
    mCellsLookupTable.resize(trkParam.CellsPerRoad() - 1);
    mCellsNeighbours.resize(trkParam.CellsPerRoad() - 1);
    mIndexTables.resize(trkParam.TrackletsPerRoad(), std::vector<int>(trkParam.ZBins * trkParam.PhiBins + 1, 0));
    mTracklets.resize(trkParam.TrackletsPerRoad());
    mTrackletsLookupTable.resize(trkParam.CellsPerRoad());
    mIndexTableUtils.setTrackingParameters(trkParam);

    std::vector<int> clsPerBin(trkParam.PhiBins * trkParam.ZBins, 0);
    for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {

      const auto& currentLayer{cl[iLayer]};
      const int clustersNum{static_cast<int>(currentLayer.size())};

      mClusters[iLayer].clear();
      mClusters[iLayer].resize(clustersNum);
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(clustersNum, false);

      std::fill(clsPerBin.begin(), clsPerBin.end(), 0);

      cHelper.clear();
      cHelper.resize(clustersNum);

      mMinR[iLayer] = 1000.f;
      mMaxR[iLayer] = -1.f;

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        const Cluster& c = currentLayer[iCluster];
        ClusterHelper& h = cHelper[iCluster];
        float x = c.xCoordinate - mPrimaryVertex.x;
        float y = c.yCoordinate - mPrimaryVertex.y;
        float phi = math_utils::calculatePhiCoordinate(x, y);
        const int zBin{mIndexTableUtils.getZBinIndex(iLayer, c.zCoordinate)};
        int bin = mIndexTableUtils.getBinIndex(zBin, mIndexTableUtils.getPhiBinIndex(phi));
        CA_DEBUGGER(assert(zBin > 0));
        h.phi = phi;
        h.r = math_utils::calculateRCoordinate(x, y);
        mMinR[iLayer] = gpu::GPUCommonMath::Min(h.r, mMinR[iLayer]);
        mMaxR[iLayer] = gpu::GPUCommonMath::Max(h.r, mMaxR[iLayer]);
        h.bin = bin;
        h.ind = clsPerBin[bin]++;
      }

      std::vector<int> lutPerBin(clsPerBin.size());
      lutPerBin[0] = 0;
      for (unsigned int iB{1}; iB < lutPerBin.size(); ++iB) {
        lutPerBin[iB] = lutPerBin[iB - 1] + clsPerBin[iB - 1];
      }

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {
        ClusterHelper& h = cHelper[iCluster];
        Cluster& c = mClusters[iLayer][lutPerBin[h.bin] + h.ind];
        c = currentLayer[iCluster];
        c.phiCoordinate = h.phi;
        c.rCoordinate = h.r;
        c.indexTableBinIndex = h.bin;
      }

      if (iLayer > 0) {
        for (unsigned int iB{0}; iB < clsPerBin.size(); ++iB) {
          mIndexTables[iLayer - 1][iB] = lutPerBin[iB];
        }
        for (auto iB{clsPerBin.size()}; iB < (int)mIndexTables[iLayer - 1].size(); iB++) {
          mIndexTables[iLayer - 1][iB] = clustersNum;
        }
      }
    }
  }

  mRoads.clear();

  for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
    if (iLayer < mCells.size()) {
      mCells[iLayer].clear();
      float cellsMemorySize =
        memParam.MemoryOffset +
        std::ceil(((memParam.CellsMemoryCoefficients[iLayer] * cl[iLayer].size()) *
                   cl[iLayer + 1].size()) *
                  cl[iLayer + 2].size());

      if (cellsMemorySize > mCells[iLayer].capacity()) {
        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if (iLayer < mCells.size() - 1) {
      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(
        std::max(cl[iLayer + 1].size(), cl[iLayer + 2].size()) +
          std::ceil((memParam.TrackletsMemoryCoefficients[iLayer + 1] *
                     cl[iLayer + 1].size()) *
                    cl[iLayer + 2].size()),
        constants::its::UnusedIndex);
      mCellsNeighbours[iLayer].clear();
    }
  }

  for (unsigned int iLayer{0}; iLayer < mClusters.size(); ++iLayer) {
    if (iLayer < mTracklets.size()) {
      mTracklets[iLayer].clear();
      float trackletsMemorySize =
        std::max(cl[iLayer].size(), cl[iLayer + 1].size()) +
        std::ceil((memParam.TrackletsMemoryCoefficients[iLayer] * cl[iLayer].size()) *
                  cl[iLayer + 1].size());

      if (trackletsMemorySize > mTracklets[iLayer].capacity()) {
        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if (iLayer < mCells.size()) {
      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(cl[iLayer + 1].size(), constants::its::UnusedIndex);
    }
  }
}

} // namespace its
} // namespace o2
