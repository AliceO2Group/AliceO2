// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///

#include <array>
#include <sstream>
#include <iostream>
#include <unistd.h>
#include <thread>

#include "DataFormatsITS/TrackITS.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"

namespace o2::its
{
constexpr int UnusedIndex{-1};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialiseHybrid(iteration, mTrkParams[iteration], nLayers);
  mTimeFrameGPU->loadTrackingFrameInfoDevice(iteration);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerTracklets(const int iteration, int, int)
{
  // if (!mTimeFrameGPU->getClusters().size()) {
  //   return;
  // }
  // const Vertex diamondVert({mTrkParams[iteration].Diamond[0], mTrkParams[iteration].Diamond[1], mTrkParams[iteration].Diamond[2]}, {25.e-6f, 0.f, 0.f, 25.e-6f, 0.f, 36.f}, 1, 1.f);
  // gsl::span<const Vertex> diamondSpan(&diamondVert, 1);
  // std::vector<std::thread> threads(mTimeFrameGPU->getNChunks());

  // for (int chunkId{0}; chunkId < mTimeFrameGPU->getNChunks(); ++chunkId) {
  //   int maxTracklets{static_cast<int>(mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->clustersPerROfCapacity) *
  //                    static_cast<int>(mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxTrackletsPerCluster)};
  //   int maxRofPerChunk{mTimeFrameGPU->mNrof / (int)mTimeFrameGPU->getNChunks()};
  //   // Define workload
  //   auto doTrackReconstruction = [&, chunkId, maxRofPerChunk, iteration]() -> void {
  //     auto offset = chunkId * maxRofPerChunk;
  //     auto maxROF = offset + maxRofPerChunk;
  //     while (offset < maxROF) {
  //       auto rofs = mTimeFrameGPU->loadChunkData<gpu::Task::Tracker>(chunkId, offset, maxROF);
  //       ////////////////////
  //       /// Tracklet finding

  //       for (int iLayer{0}; iLayer < nLayers - 1; ++iLayer) {
  //         auto nclus = mTimeFrameGPU->getTotalClustersPerROFrange(offset, rofs, iLayer);
  //         const float meanDeltaR{mTrkParams[iteration].LayerRadii[iLayer + 1] - mTrkParams[iteration].LayerRadii[iLayer]};
  //         gpu::computeLayerTrackletsKernelMultipleRof<<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
  //           iLayer,                                                                                // const int layerIndex,
  //           iteration,                                                                             // const int iteration,
  //           offset,                                                                                // const unsigned int startRofId,
  //           rofs,                                                                                  // const unsigned int rofSize,
  //           0,                                                                                     // const unsigned int deltaRof,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(iLayer),                            // const Cluster* clustersCurrentLayer,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(iLayer + 1),                        // const Cluster* clustersNextLayer,
  //           mTimeFrameGPU->getDeviceROframesClusters(iLayer),                                      // const int* roFrameClustersCurrentLayer, // Number of clusters on layer 0 per ROF
  //           mTimeFrameGPU->getDeviceROframesClusters(iLayer + 1),                                  // const int* roFrameClustersNextLayer,    // Number of clusters on layer 1 per ROF
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceIndexTables(iLayer + 1),                     // const int* indexTableNextLayer,
  //           mTimeFrameGPU->getDeviceUsedClusters(iLayer),                                          // const int* usedClustersCurrentLayer,
  //           mTimeFrameGPU->getDeviceUsedClusters(iLayer + 1),                                      // const int* usedClustersNextLayer,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),                           // Tracklet* tracklets,       // output data
  //           mTimeFrameGPU->getDeviceVertices(),                                                    // const Vertex* vertices,
  //           mTimeFrameGPU->getDeviceROframesPV(),                                                  // const int* pvROFrame,
  //           mTimeFrameGPU->getPhiCut(iLayer),                                                      // const float phiCut,
  //           mTimeFrameGPU->getMinR(iLayer + 1),                                                    // const float minR,
  //           mTimeFrameGPU->getMaxR(iLayer + 1),                                                    // const float maxR,
  //           meanDeltaR,                                                                            // const float meanDeltaR,
  //           mTimeFrameGPU->getPositionResolution(iLayer),                                          // const float positionResolution,
  //           mTimeFrameGPU->getMSangle(iLayer),                                                     // const float mSAngle,
  //           mTimeFrameGPU->getDeviceTrackingParameters(),                                          // const StaticTrackingParameters<nLayers>* trkPars,
  //           mTimeFrameGPU->getDeviceIndexTableUtils(),                                             // const IndexTableUtils* utils
  //           mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->clustersPerROfCapacity,  // const int clustersPerROfCapacity,
  //           mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxTrackletsPerCluster); // const int maxTrackletsPerCluster

  //         // Remove empty tracklets due to striding.
  //         auto nulltracklet = o2::its::Tracklet{};
  //         auto thrustTrackletsBegin = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer));
  //         auto thrustTrackletsEnd = thrust::device_ptr<o2::its::Tracklet>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer) + (int)rofs * maxTracklets);
  //         auto thrustTrackletsAfterEraseEnd = thrust::remove(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()),
  //                                                            thrustTrackletsBegin,
  //                                                            thrustTrackletsEnd,
  //                                                            nulltracklet);
  //         // Sort tracklets by first cluster index.
  //         thrust::sort(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()),
  //                      thrustTrackletsBegin,
  //                      thrustTrackletsAfterEraseEnd,
  //                      gpu::trackletSortIndexFunctor<o2::its::Tracklet>());

  //         // Remove duplicates.
  //         auto thrustTrackletsAfterUniqueEnd = thrust::unique(THRUST_NAMESPACE::par.on(mTimeFrameGPU->getStream(chunkId).get()), thrustTrackletsBegin, thrustTrackletsAfterEraseEnd);

  //         discardResult(cudaStreamSynchronize(mTimeFrameGPU->getStream(chunkId).get()));
  //         mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer] = thrustTrackletsAfterUniqueEnd - thrustTrackletsBegin;
  //         // Compute tracklet lookup table.
  //         gpu::compileTrackletsLookupTableKernel<<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
  //                                                                                                            mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),
  //                                                                                                            mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer]);
  //         discardResult(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
  //                                                     mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),        // d_in
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer),        // d_out
  //                                                     nclus,                                                                          // num_items
  //                                                     mTimeFrameGPU->getStream(chunkId).get()));

  //         // Create tracklets labels, at the moment on the host
  //         if (mTimeFrameGPU->hasMCinformation()) {
  //           std::vector<o2::its::Tracklet> tracklets(mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer]);
  //           checkGPUError(cudaHostRegister(tracklets.data(), tracklets.size() * sizeof(o2::its::Tracklet), cudaHostRegisterDefault));
  //           checkGPUError(cudaMemcpyAsync(tracklets.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer), tracklets.size() * sizeof(o2::its::Tracklet), cudaMemcpyDeviceToHost, mTimeFrameGPU->getStream(chunkId).get()));
  //           for (auto& trk : tracklets) {
  //             MCCompLabel label;
  //             int currentId{mTimeFrameGPU->mClusters[iLayer][trk.firstClusterIndex].clusterId};   // This is not yet offsetted to the index of the first cluster of the chunk
  //             int nextId{mTimeFrameGPU->mClusters[iLayer + 1][trk.secondClusterIndex].clusterId}; // This is not yet offsetted to the index of the first cluster of the chunk
  //             for (auto& lab1 : mTimeFrameGPU->getClusterLabels(iLayer, currentId)) {
  //               for (auto& lab2 : mTimeFrameGPU->getClusterLabels(iLayer + 1, nextId)) {
  //                 if (lab1 == lab2 && lab1.isValid()) {
  //                   label = lab1;
  //                   break;
  //                 }
  //               }
  //               if (label.isValid()) {
  //                 break;
  //               }
  //             }
  //             // TODO: implment label merging.
  //             // mTimeFrameGPU->getTrackletsLabel(iLayer).emplace_back(label);
  //           }
  //           checkGPUError(cudaHostUnregister(tracklets.data()));
  //         }
  //       }

  //       ////////////////
  //       /// Cell finding
  //       for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
  //         // Compute layer cells.
  //         gpu::computeLayerCellsKernel<true><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer + 1),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer + 1),
  //           mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],
  //           nullptr,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),
  //           mTimeFrameGPU->getDeviceTrackingParameters());

  //         // Compute number of found Cells
  //         checkGPUError(cub::DeviceReduce::Sum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
  //                                              mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
  //                                              mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_in
  //                                              mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells() + iLayer,               // d_out
  //                                              mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],                              // num_items
  //                                              mTimeFrameGPU->getStream(chunkId).get()));
  //         // Compute LUT
  //         discardResult(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
  //                                                     mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_in
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),            // d_out
  //                                                     mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],                              // num_items
  //                                                     mTimeFrameGPU->getStream(chunkId).get()));

  //         gpu::computeLayerCellsKernel<false><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(iLayer + 1),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceTrackletsLookupTables(iLayer + 1),
  //           mTimeFrameGPU->getHostNTracklets(chunkId)[iLayer],
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer),
  //           mTimeFrameGPU->getDeviceTrackingParameters());
  //       }
  //       checkGPUError(cudaMemcpyAsync(mTimeFrameGPU->getHostNCells(chunkId).data(),
  //                                     mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
  //                                     (nLayers - 2) * sizeof(int),
  //                                     cudaMemcpyDeviceToHost,
  //                                     mTimeFrameGPU->getStream(chunkId).get()));

  //       // Create cells labels
  //       // TODO: make it work after fixing the tracklets labels
  //       if (mTimeFrameGPU->hasMCinformation()) {
  //         for (int iLayer{0}; iLayer < nLayers - 2; ++iLayer) {
  //           std::vector<o2::its::Cell> cells(mTimeFrameGPU->getHostNCells(chunkId)[iLayer]);
  //           // Async with not registered memory?
  //           checkGPUError(cudaMemcpyAsync(cells.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer), mTimeFrameGPU->getHostNCells(chunkId)[iLayer] * sizeof(o2::its::Cell), cudaMemcpyDeviceToHost));
  //           for (auto& cell : cells) {
  //             MCCompLabel currentLab{mTimeFrameGPU->getTrackletsLabel(iLayer)[cell.getFirstTrackletIndex()]};
  //             MCCompLabel nextLab{mTimeFrameGPU->getTrackletsLabel(iLayer + 1)[cell.getSecondTrackletIndex()]};
  //             mTimeFrameGPU->getCellsLabel(iLayer).emplace_back(currentLab == nextLab ? currentLab : MCCompLabel());
  //           }
  //         }
  //       }

  //       /////////////////////
  //       /// Neighbour finding
  //       for (int iLayer{0}; iLayer < nLayers - 3; ++iLayer) {
  //         gpu::computeLayerCellNeighboursKernel<true><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer + 1),
  //           iLayer,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer + 1),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),
  //           nullptr,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
  //           mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize);

  //         // Compute Cell Neighbours LUT
  //         checkGPUError(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
  //                                                     mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_in
  //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_out
  //                                                     mTimeFrameGPU->getHostNCells(chunkId)[iLayer + 1],                              // num_items
  //                                                     mTimeFrameGPU->getStream(chunkId).get()));

  //         gpu::computeLayerCellNeighboursKernel<false><<<10, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCells(iLayer + 1),
  //           iLayer,
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellsLookupTables(iLayer + 1),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeighbours(iLayer),
  //           mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),
  //           mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize);

  //         // if (!chunkId) {
  //         //   gpu::printBufferLayerOnThread<<<1, 1, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(iLayer,
  //         //                                                                                       mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeighbours(iLayer),
  //         //                                                                                       mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->maxNeighboursSize * rofs);
  //         // }
  //       }
  //       // Download cells into vectors

  //       for (int iLevel{nLayers - 2}; iLevel >= mTrkParams[iteration].CellMinimumLevel(); --iLevel) {
  //         const int minimumLevel{iLevel - 1};
  //         for (int iLayer{nLayers - 3}; iLayer >= minimumLevel; --iLayer) {
  //           // gpu::computeLayerRoadsKernel<true><<<1, 1, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(iLevel,                                                               // const int level,
  //           //  iLayer,                                                               // const int layerIndex,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayCells(),               // const CellSeed** cells,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundCells(),              // const int* nCells,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayNeighboursCell(),      // const int** neighbours,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceArrayNeighboursCellLUT(),   // const int** neighboursLUT,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceRoads(),                    // Road* roads,
  //           //  mTimeFrameGPU->getChunk(chunkId).getDeviceRoadsLookupTables(iLayer)); // int* roadsLookupTable
  //         }
  //       }

  //       // End of tracking for this chunk
  //       offset += rofs;
  //     }
  //   };
  //   threads[chunkId] = std::thread(doTrackReconstruction);
  // }
  // for (auto& thread : threads) {
  //   thread.join();
  // }

  // mTimeFrameGPU->wipe(nLayers);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerCells(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::extendTracks(const int iteration)
{
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::setBz(float bz)
{
  mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfTracklets() const
{
  return mTimeFrameGPU->getNumberOfTracklets();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

////////////////////////////////////////////////////////////////////////////////
// Hybrid tracking
template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeTrackletsHybrid(const int iteration, int iROFslice, int iVertex)
{
  TrackerTraits::computeLayerTracklets(iteration, iROFslice, iVertex);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeCellsHybrid(const int iteration)
{
  TrackerTraits::computeLayerCells(iteration);
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighboursHybrid(const int iteration)
{
  TrackerTraits::findCellsNeighbours(iteration);
  // for (int iLayer{0}; iLayer < mTrkParams[iteration].CellsPerRoad() - 1; ++iLayer) {
  //   const int nextLayerCellsNum{static_cast<int>(mTimeFrameGPU->getCells()[iLayer + 1].size())};
  //   mTimeFrameGPU->getCellsNeighboursLUT()[iLayer].clear();
  //   mTimeFrameGPU->getCellsNeighboursLUT()[iLayer].resize(nextLayerCellsNum, 0);
  //   if (mTimeFrameGPU->getCells()[iLayer + 1].empty() ||
  //       mTimeFrameGPU->getCellsLookupTable()[iLayer].empty()) {
  //     mTimeFrameGPU->getCellsNeighbours()[iLayer].clear();
  //     continue;
  //   }

  //   int layerCellsNum{static_cast<int>(mTimeFrameGPU->getCells()[iLayer].size())};
  //   std::vector<std::pair<int, int>> cellsNeighbours;
  //   cellsNeighbours.reserve(nextLayerCellsNum);
  //   mTimeFrameGPU->createCellNeighboursDevice(iLayer, cellsNeighbours);

  //   // // [...]
  //   // cellNeighboursHandler<true>(mTimeFrameGPU->getDeviceNeighbours(iLayer));
  //   // //         // Compute Cell Neighbours LUT
  //   // //         checkGPUError(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),                       // d_temp_storage
  //   // //                                                     mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize, // temp_storage_bytes
  //   // //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_in
  //   // //                                                     mTimeFrameGPU->getChunk(chunkId).getDeviceCellNeigboursLookupTables(iLayer),    // d_out
  //   // //                                                     mTimeFrameGPU->getHostNCells(chunkId)[iLayer + 1],                              // num_items
  //   // //                                                     mTimeFrameGPU->getStream(chunkId).get()));

  //   // cellsNeighboursHandler<false>(mTimeFrameGPU->getDeviceNeighbours(iLayer));
  //   // // [...]

  //   std::sort(cellsNeighbours.begin(), cellsNeighbours.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
  //     return a.second < b.second;
  //   });
  //   mTimeFrameGPU->getCellsNeighbours()[iLayer].clear();
  //   mTimeFrameGPU->getCellsNeighbours()[iLayer].reserve(cellsNeighbours.size());
  //   for (auto& cellNeighboursIndex : cellsNeighbours) {
  //     mTimeFrameGPU->getCellsNeighbours()[iLayer].push_back(cellNeighboursIndex.first);
  //   }
  //   std::inclusive_scan(mTimeFrameGPU->getCellsNeighboursLUT()[iLayer].begin(), mTimeFrameGPU->getCellsNeighboursLUT()[iLayer].end(), mTimeFrameGPU->getCellsNeighboursLUT()[iLayer].begin());
  // }
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration)
{
  for (int startLevel{mTrkParams[iteration].CellsPerRoad()}; startLevel >= mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    std::vector<CellSeed> trackSeeds;
    for (int startLayer{mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      std::vector<int> lastCellId, updatedCellId;
      std::vector<CellSeed> lastCellSeed, updatedCellSeed;

      processNeighbours(startLayer, startLevel, mTimeFrame->getCells()[startLayer], lastCellId, updatedCellSeed, updatedCellId);

      int level = startLevel;
      for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
        lastCellSeed.swap(updatedCellSeed);
        lastCellId.swap(updatedCellId);
        std::vector<CellSeed>().swap(updatedCellSeed); /// tame the memory peaks
        updatedCellId.clear();
        processNeighbours(iLayer, --level, lastCellSeed, lastCellId, updatedCellSeed, updatedCellId);
      }
      for (auto& seed : updatedCellSeed) {
        if (seed.getQ2Pt() > 1.e3 || seed.getChi2() > mTrkParams[0].MaxChi2NDF * ((startLevel + 2) * 2 - 5)) {
          continue;
        }
        trackSeeds.push_back(seed);
      }
    }
    if (!trackSeeds.size()) {
      LOGP(info, "No track seeds found, skipping track finding");
      continue;
    }
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds);
    mTimeFrameGPU->loadTrackSeedsDevice(trackSeeds);

    trackSeedHandler(
      mTimeFrameGPU->getDeviceTrackSeeds(),             // CellSeed* trackSeeds,
      mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo** foundTrackingFrameInfo,
      mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt* tracks,
      trackSeeds.size(),                                // const size_t nSeeds,
      mBz,                                              // const float Bz,
      startLevel,                                       // const int startLevel,
      mTrkParams[0].MaxChi2ClusterAttachment,           // float maxChi2ClusterAttachment,
      mTrkParams[0].MaxChi2NDF,                         // float maxChi2NDF,
      mTimeFrameGPU->getDevicePropagator(),             // const o2::base::Propagator* propagator
      mCorrType);                                       // o2::base::PropagatorImpl<float>::MatCorrType

    mTimeFrameGPU->downloadTrackITSExtDevice(trackSeeds);

    auto& tracks = mTimeFrameGPU->getTrackITSExt();
    std::sort(tracks.begin(), tracks.end(), [](const TrackITSExt& a, const TrackITSExt& b) {
      return a.getChi2() < b.getChi2();
    });

    for (auto& track : tracks) {
      if (!track.getChi2()) {
        continue; // this is to skip the unset tracks that are put at the beginning of the vector by the sorting. To see if this can be optimised.
      }
      int nShared = 0;
      bool isFirstShared{false};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == UnusedIndex) {
          continue;
        }
        nShared += int(mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
        isFirstShared |= !iLayer && mTimeFrame->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
      }

      if (nShared > mTrkParams[0].ClusterSharing) {
        continue;
      }

      std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
      for (int iLayer{0}; iLayer < mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == UnusedIndex) {
          continue;
        }
        mTimeFrame->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        int currentROF = mTimeFrame->getClusterROF(iLayer, track.getClusterIndex(iLayer));
        for (int iR{0}; iR < 3; ++iR) {
          if (rofs[iR] == INT_MAX) {
            rofs[iR] = currentROF;
          }
          if (rofs[iR] == currentROF) {
            break;
          }
        }
      }
      if (rofs[2] != INT_MAX) {
        continue;
      }
      if (rofs[1] != INT_MAX) {
        track.setNextROFbit();
      }
      mTimeFrame->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
    }
  }
  if (iteration == mTrkParams.size() - 1) {
    mTimeFrameGPU->unregisterHostMemory(0);
  }
};

template class TrackerTraitsGPU<7>;
} // namespace o2::its
