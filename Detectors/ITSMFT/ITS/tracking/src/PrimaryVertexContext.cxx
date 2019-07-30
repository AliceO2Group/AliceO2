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
      mClusters[iLayer].resize(clustersNum);
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(clustersNum, false);

      constexpr int _size = constants::index_table::PhiBins * constants::index_table::ZBins;
      std::array<int, _size> clsPerBin;
      std::array<int, _size> lutPerBin;
      for (int iB{0}; iB < _size; ++iB) {
        clsPerBin[iB] = 0;
        lutPerBin[iB] = 0;
      }

      std::vector<int> bins(clustersNum);
      for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        const float& x = currentLayer.at(iCluster).xCoordinate;
        const float& y = currentLayer.at(iCluster).yCoordinate;
        const float& z = currentLayer.at(iCluster).zCoordinate;
        const float phi =  math_utils::calculatePhiCoordinate(x - pVtx[0], y - pVtx[1]);
        int bin = index_table_utils::getBinIndex(index_table_utils::getZBinIndex(iLayer, z),
                                                 index_table_utils::getPhiBinIndex(phi));
        bins[iCluster] = bin;
        clsPerBin[bin]++;
      }

      for (int iB{1}; iB < _size; ++iB) {
        lutPerBin[iB] = lutPerBin[iB - 1] + clsPerBin[iB - 1];
      }

      for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        const int& bin = bins[iCluster];
        mClusters[iLayer][lutPerBin[bin]].Init(iLayer, mPrimaryVertex, currentLayer.at(iCluster));
        lutPerBin[bin]++;
      }

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
