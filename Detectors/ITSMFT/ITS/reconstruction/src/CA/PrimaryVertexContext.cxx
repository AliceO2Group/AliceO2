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

#include "ITSReconstruction/CA/PrimaryVertexContext.h"

#include "ITSReconstruction/CA/Event.h"

namespace o2
{
namespace ITS
{
namespace CA
{

PrimaryVertexContext::PrimaryVertexContext()
{
  // Nothing to do
}

void PrimaryVertexContext::initialise(const Event& event, const int primaryVertexIndex)
{
  mPrimaryVertex = event.getPrimaryVertex(primaryVertexIndex);

  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const Layer& currentLayer{ event.getLayer(iLayer) };
    const int clustersNum{ currentLayer.getClustersSize() };

    mClusters[iLayer].clear();
    mClusters[iLayer].reserve(clustersNum);
    mUsedClusters[iLayer].clear();
    mUsedClusters[iLayer].resize(clustersNum, false);

    for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {

      const Cluster& currentCluster{ currentLayer.getCluster(iCluster) };
      mClusters[iLayer].emplace_back(iLayer, event.getPrimaryVertex(primaryVertexIndex), currentCluster);
    }

    std::sort(mClusters[iLayer].begin(), mClusters[iLayer].end(), [](Cluster& cluster1, Cluster& cluster2) {
      return cluster1.indexTableBinIndex < cluster2.indexTableBinIndex;
    });

    if (iLayer < Constants::ITS::CellsPerRoad) {

      mCells[iLayer].clear();
      float cellsMemorySize =
        Constants::Memory::Offset +
        std::ceil(((Constants::Memory::CellsMemoryCoefficients[iLayer] *
        event.getLayer(iLayer).getClustersSize()) *
        event.getLayer(iLayer + 1).getClustersSize()) *
        event.getLayer(iLayer + 2).getClustersSize());

      if (cellsMemorySize > mCells[iLayer].capacity()) {
        mCells[iLayer].reserve(cellsMemorySize);
      }
    }

    if (iLayer < Constants::ITS::CellsPerRoad - 1) {

      mCellsLookupTable[iLayer].clear();
      mCellsLookupTable[iLayer].resize(std::max(event.getLayer(iLayer + 1).getClustersSize(),
                                                event.getLayer(iLayer + 2).getClustersSize()) +
                                       std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer + 1] *
                                       event.getLayer(iLayer + 1).getClustersSize()) *
                                       event.getLayer(iLayer + 2).getClustersSize()),
                                       Constants::ITS::UnusedIndex);

      mCellsNeighbours[iLayer].clear();
    }
  }

  mRoads.clear();
  mTracks.clear();
  mTrackLabels.clear();

#if TRACKINGITSU_GPU_MODE
  mGPUContextDevicePointer = mGPUContext.initialize(mPrimaryVertex, mClusters, mCells, mCellsLookupTable);
#else
  for (int iLayer{ 0 }; iLayer < Constants::ITS::LayersNumber; ++iLayer) {

    const int clustersNum = static_cast<int>(mClusters[iLayer].size());

    if (iLayer > 0) {

      int previousBinIndex{ 0 };
      mIndexTables[iLayer - 1][0] = 0;

      for (int iCluster{ 0 }; iCluster < clustersNum; ++iCluster) {

        const int currentBinIndex{ mClusters[iLayer][iCluster].indexTableBinIndex };

        if (currentBinIndex > previousBinIndex) {

          for (int iBin{ previousBinIndex + 1 }; iBin <= currentBinIndex; ++iBin) {

            mIndexTables[iLayer - 1][iBin] = iCluster;
          }

          previousBinIndex = currentBinIndex;
        }
      }

      for (int iBin{ previousBinIndex + 1 }; iBin <= Constants::IndexTable::ZBins * Constants::IndexTable::PhiBins;
           iBin++) {

        mIndexTables[iLayer - 1][iBin] = clustersNum;
      }
    }

    if (iLayer < Constants::ITS::TrackletsPerRoad) {

      mTracklets[iLayer].clear();

      float trackletsMemorySize =
        std::max(event.getLayer(iLayer).getClustersSize(), event.getLayer(iLayer + 1).getClustersSize()) +
        std::ceil((Constants::Memory::TrackletsMemoryCoefficients[iLayer] *
                  event.getLayer(iLayer).getClustersSize()) *
                  event.getLayer(iLayer + 1).getClustersSize());

      if (trackletsMemorySize > mTracklets[iLayer].capacity()) {
        mTracklets[iLayer].reserve(trackletsMemorySize);
      }
    }

    if (iLayer < Constants::ITS::CellsPerRoad) {

      mTrackletsLookupTable[iLayer].clear();
      mTrackletsLookupTable[iLayer].resize(event.getLayer(iLayer + 1).getClustersSize(), Constants::ITS::UnusedIndex);
    }
  }
#endif
}
}
}
}
