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

  struct ClusterHelper {
    float phi;
    float r;
    int bin;
    int ind;
  };

  mPrimaryVertex = { pVtx[0], pVtx[1], pVtx[2] };

  if (iteration == 0) {

    std::vector<ClusterHelper> cHelper;

    for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {

      const auto& currentLayer{ cl[iLayer] };
      const int clustersNum{ static_cast<int>(currentLayer.size()) };

      mClusters[iLayer].clear();
      mClusters[iLayer].resize(clustersNum);
      mUsedClusters[iLayer].clear();
      mUsedClusters[iLayer].resize(clustersNum, false);

      constexpr int _size = constants::index_table::PhiBins * constants::index_table::ZBins;
      std::array<int, _size> clsPerBin;
      for (int iB{0}; iB < _size; ++iB) {
        clsPerBin[iB] = 0;
      }

      cHelper.clear();
      cHelper.resize(clustersNum);

      for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        const Cluster& c = currentLayer[iCluster];
        ClusterHelper& h = cHelper[iCluster];
        float x = c.xCoordinate - mPrimaryVertex.x;
        float y = c.yCoordinate - mPrimaryVertex.y;
        float phi = math_utils::calculatePhiCoordinate(x, y);
        int bin = index_table_utils::getBinIndex(index_table_utils::getZBinIndex(iLayer, c.zCoordinate),
                                                 index_table_utils::getPhiBinIndex(phi));
        h.phi = phi;
        h.r = math_utils::calculateRCoordinate(x, y);
        h.bin = bin;
        h.ind = clsPerBin[bin]++;
      }

      std::array<int, _size> lutPerBin;
      lutPerBin[0] = 0;
      for (int iB{1}; iB < _size; ++iB) {
        lutPerBin[iB] = lutPerBin[iB - 1] + clsPerBin[iB - 1];
      }

      for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {
        ClusterHelper& h = cHelper[iCluster];
        Cluster& c = mClusters[iLayer][lutPerBin[h.bin] + h.ind];
        c = currentLayer[iCluster];
        c.phiCoordinate = h.phi;
        c.rCoordinate = h.r;
        c.indexTableBinIndex = h.bin;
      }

      if (iLayer > 0) {
        for (int iB{ 0 }; iB < _size; ++iB) {
          mIndexTables[iLayer - 1][iB] = lutPerBin[iB];
        }
        for (int iB{ _size }; iB < (int)mIndexTables[iLayer - 1].size(); iB++) {
          mIndexTables[iLayer - 1][iB] = clustersNum;
        }
      }
    }
  }

  mRoads.clear();

  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {
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

  for (int iLayer{ 0 }; iLayer < constants::its::LayersNumber; ++iLayer) {
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
