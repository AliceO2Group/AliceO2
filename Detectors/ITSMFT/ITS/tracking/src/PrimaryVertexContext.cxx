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

void PrimaryVertexContext::initialise(const MemoryParameters& memParam, const std::array<std::vector<Cluster>, constants::its::LayersNumber>& cl,
                                      const std::array<float, 3>& pVtx, const int iteration)
{
  mPrimaryVertex = {pVtx[0], pVtx[1], pVtx[2]};

  for (int iLayer{0}; iLayer < constants::its::LayersNumber; ++iLayer) {

    const auto& currentLayer{cl[iLayer]};
    const int clustersNum{static_cast<int>(currentLayer.size())};

    if (iteration == 0) {
      mClusters[iLayer].clear();
      mClusters[iLayer].reserve(clustersNum);
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(clustersNum, false);

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {

        const Cluster& currentCluster{currentLayer.at(iCluster)};
        mClusters[iLayer].emplace_back(iLayer, mPrimaryVertex, currentCluster);
      }

      std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
        return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
      });
    }

    if (iLayer < constants::its::CellsPerRoad) {

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

    if (iLayer < constants::its::CellsPerRoad - 1) {

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

  mRoads.clear();

  for (int iLayer{0}; iLayer < constants::its::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(mClusters[iLayer].size());

    if (iLayer > 0 && iteration == 0) {

      int previousBinIndex{0};
      mIndexTables[iLayer - 1][0] = 0;

      for (int iCluster{0}; iCluster < clustersNum; ++iCluster) {

        const int currentBinIndex{mClusters[iLayer][iCluster].indexTableBinIndex};

        if (currentBinIndex > previousBinIndex) {

          for (int iBin{previousBinIndex + 1}; iBin <= currentBinIndex; ++iBin) {

            mIndexTables[iLayer - 1][iBin] = iCluster;
          }

          previousBinIndex = currentBinIndex;
        }
      }

      for (int iBin{previousBinIndex + 1}; iBin < (int)mIndexTables[iLayer - 1].size(); iBin++) {
        mIndexTables[iLayer - 1][iBin] = clustersNum;
      }
    }

    if (iLayer < constants::its::TrackletsPerRoad) {

      mTracklets[iLayer].clear();

      float trackletsMemorySize =
        std::max(cl[iLayer].size(), cl[iLayer + 1].size()) +
        std::ceil((memParam.TrackletsMemoryCoefficients[iLayer] * cl[iLayer].size()) *
                  cl[iLayer + 1].size());

      if (trackletsMemorySize > mTracklets[iLayer].capacity()) {
        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if (iLayer < constants::its::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(cl[iLayer + 1].size(), constants::its::UnusedIndex);
    }
  }
}

} // namespace its
} // namespace o2
