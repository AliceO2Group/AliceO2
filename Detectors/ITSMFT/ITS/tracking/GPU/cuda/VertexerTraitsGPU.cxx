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
//
/// \author matteo.concas@cern.ch

#include <iostream>
#include <sstream>
#include <fstream>
#include <array>
#include <cassert>
#include <thread>

#ifdef VTX_DEBUG
#include "TTree.h"
#include "TFile.h"
#endif

#include "ITStrackingGPU/VertexingKernels.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"

namespace o2::its
{
VertexerTraitsGPU::VertexerTraitsGPU()
{
  setIsGPU(true);
}

void VertexerTraitsGPU::initialise(const TrackingParameters& trackingParams, const int iteration)
{
  mTimeFrameGPU->initialise(0, trackingParams, 3, &mIndexTableUtils, &mTfGPUParams);
}
void VertexerTraitsGPU::updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar, const TimeFrameGPUParameters& tfPar)
{
  mVrtParams = vrtPar;
  mTfGPUParams = tfPar;
  mIndexTableUtils.setTrackingParameters(vrtPar[0]);
  for (auto& par : mVrtParams) {
    par.phiSpan = static_cast<int>(std::ceil(mIndexTableUtils.getNphiBins() * par.phiCut / constants::math::TwoPi));
    par.zSpan = static_cast<int>(std::ceil(par.zCut * mIndexTableUtils.getInverseZCoordinate(0)));
  }
}

void VertexerTraitsGPU::computeTracklets(const int iteration)
{
  if (!mTimeFrameGPU->getClusters().size()) {
    return;
  }
  std::vector<std::thread> threads(mTimeFrameGPU->getNChunks());
  for (int chunkId{0}; chunkId < mTimeFrameGPU->getNChunks(); ++chunkId) {
    //   int rofPerChunk{mTimeFrameGPU->mNrof / (int)mTimeFrameGPU->getNChunks()};
    //   mTimeFrameGPU->getVerticesInChunks()[chunkId].clear();
    //   mTimeFrameGPU->getNVerticesInChunks()[chunkId].clear();
    //   mTimeFrameGPU->getLabelsInChunks()[chunkId].clear();
    //   auto doVertexReconstruction = [&, chunkId, rofPerChunk]() -> void {
    //     auto offset = chunkId * rofPerChunk;
    //     auto maxROF = offset + rofPerChunk;
    //     while (offset < maxROF) {
    //       auto rofs = mTimeFrameGPU->loadChunkData<gpu::Task::Vertexer>(chunkId, offset, maxROF);
    //       RANGE("chunk_gpu_vertexing", 1);
    //       // gpu::GpuTimer timer{offset, mTimeFrameGPU->getStream(chunkId).get()};
    //       // timer.Start("vtTrackletFinder");
    //       gpu::trackleterKernelMultipleRof<TrackletMode::Layer0Layer1><<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(0),         // const Cluster* clustersNextLayer,    // 0 2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(1),         // const Cluster* clustersCurrentLayer, // 1 1
    //         mTimeFrameGPU->getDeviceROframesClusters(0),                   // const int* sizeNextLClusters,
    //         mTimeFrameGPU->getDeviceROframesClusters(1),                   // const int* sizeCurrentLClusters,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceIndexTables(0),      // const int* nextIndexTables,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(0),        // Tracklet* Tracklets,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(0), // int* foundTracklets,
    //         mTimeFrameGPU->getDeviceIndexTableUtils(),                     // const IndexTableUtils* utils,
    //         offset,                                                        // const unsigned int startRofId,
    //         rofs,                                                          // const unsigned int rofSize,
    //         mVrtParams.phiCut,                                             // const float phiCut,
    //         mVrtParams.maxTrackletsPerCluster);                            // const size_t maxTrackletsPerCluster = 1e2

    //       gpu::trackleterKernelMultipleRof<TrackletMode::Layer1Layer2><<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(2),         // const Cluster* clustersNextLayer,    // 0 2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(1),         // const Cluster* clustersCurrentLayer, // 1 1
    //         mTimeFrameGPU->getDeviceROframesClusters(2),                   // const int* sizeNextLClusters,
    //         mTimeFrameGPU->getDeviceROframesClusters(1),                   // const int* sizeCurrentLClusters,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceIndexTables(2),      // const int* nextIndexTables,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(1),        // Tracklet* Tracklets,
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(1), // int* foundTracklets,
    //         mTimeFrameGPU->getDeviceIndexTableUtils(),                     // const IndexTableUtils* utils,
    //         offset,                                                        // const unsigned int startRofId,
    //         rofs,                                                          // const unsigned int rofSize,
    //         mVrtParams.phiCut,                                             // const float phiCut,
    //         mVrtParams.maxTrackletsPerCluster);                            // const size_t maxTrackletsPerCluster = 1e2

    //       gpu::trackletSelectionKernelMultipleRof<true><<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(0),            // const Cluster* clusters0,               // Clusters on layer 0
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(1),            // const Cluster* clusters1,               // Clusters on layer 1
    //         mTimeFrameGPU->getDeviceROframesClusters(0),                      // const int* sizeClustersL0,              // Number of clusters on layer 0 per ROF
    //         mTimeFrameGPU->getDeviceROframesClusters(1),                      // const int* sizeClustersL1,              // Number of clusters on layer 1 per ROF
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(0),           // Tracklet* tracklets01,                  // Tracklets on layer 0-1
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(1),           // Tracklet* tracklets12,                  // Tracklets on layer 1-2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(0),    // const int* nFoundTracklets01,           // Number of tracklets found on layers 0-1
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(1),    // const int* nFoundTracklet12,            // Number of tracklets found on layers 1-2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceUsedTracklets(),        // unsigned char* usedTracklets,           // Used tracklets
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceLines(),                // Line* lines,                            // Lines
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundLines(),          // int* nFoundLines,                       // Number of found lines
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNExclusiveFoundLines(), // int* nExclusiveFoundLines,              // Number of found lines exclusive scan
    //         offset,                                                           // const unsigned int startRofId,          // Starting ROF ID
    //         rofs,                                                             // const unsigned int rofSize,             // Number of ROFs to consider
    //         mVrtParams.maxTrackletsPerCluster,                                // const int maxTrackletsPerCluster = 1e2, // Maximum number of tracklets per cluster
    //         mVrtParams.tanLambdaCut,                                          // const float tanLambdaCut = 0.025f,      // Cut on tan lambda
    //         mVrtParams.phiCut);                                               // const float phiCut = 0.002f)            // Cut on phi

    //       discardResult(cub::DeviceScan::ExclusiveSum(mTimeFrameGPU->getChunk(chunkId).getDeviceCUBTmpBuffer(),
    //                                                   mTimeFrameGPU->getChunk(chunkId).getTimeFrameGPUParameters()->tmpCUBBufferSize,
    //                                                   mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundLines(),
    //                                                   mTimeFrameGPU->getChunk(chunkId).getDeviceNExclusiveFoundLines(),
    //                                                   mTimeFrameGPU->getTotalClustersPerROFrange(offset, rofs, 1),
    //                                                   mTimeFrameGPU->getStream(chunkId).get()));

    //       // Reset used tracklets
    //       checkGPUError(cudaMemsetAsync(mTimeFrameGPU->getChunk(chunkId).getDeviceUsedTracklets(),
    //                                     false,
    //                                     sizeof(unsigned char) * mVrtParams.maxTrackletsPerCluster * mTimeFrameGPU->getTotalClustersPerROFrange(offset, rofs, 1),
    //                                     mTimeFrameGPU->getStream(chunkId).get()),
    //                     __FILE__, __LINE__);

    //       gpu::trackletSelectionKernelMultipleRof<false><<<rofs, 1024, 0, mTimeFrameGPU->getStream(chunkId).get()>>>(
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(0),            // const Cluster* clusters0,               // Clusters on layer 0
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceClusters(1),            // const Cluster* clusters1,               // Clusters on layer 1
    //         mTimeFrameGPU->getDeviceROframesClusters(0),                      // const int* sizeClustersL0,              // Number of clusters on layer 0 per ROF
    //         mTimeFrameGPU->getDeviceROframesClusters(1),                      // const int* sizeClustersL1,              // Number of clusters on layer 1 per ROF
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(0),           // Tracklet* tracklets01,                  // Tracklets on layer 0-1
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceTracklets(1),           // Tracklet* tracklets12,                  // Tracklets on layer 1-2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(0),    // const int* nFoundTracklets01,           // Number of tracklets found on layers 0-1
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNTrackletCluster(1),    // const int* nFoundTracklet12,            // Number of tracklets found on layers 1-2
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceUsedTracklets(),        // unsigned char* usedTracklets,           // Used tracklets
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceLines(),                // Line* lines,                            // Lines
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundLines(),          // int* nFoundLines,                       // Number of found lines
    //         mTimeFrameGPU->getChunk(chunkId).getDeviceNExclusiveFoundLines(), // int* nExclusiveFoundLines,              // Number of found lines exclusive scan
    //         offset,                                                           // const unsigned int startRofId,          // Starting ROF ID
    //         rofs,                                                             // const unsigned int rofSize,             // Number of ROFs to consider
    //         mVrtParams.maxTrackletsPerCluster,                                // const int maxTrackletsPerCluster = 1e2, // Maximum number of tracklets per cluster
    //         mVrtParams.tanLambdaCut,                                          // const float tanLambdaCut = 0.025f,      // Cut on tan lambda
    //         mVrtParams.phiCut);                                               // const float phiCut = 0.002f)            // Cut on phi

    //       int nClusters = mTimeFrameGPU->getTotalClustersPerROFrange(offset, rofs, 1);
    //       int lastFoundLines;
    //       std::vector<int> exclusiveFoundLinesHost(nClusters + 1);

    //       // Obtain whole exclusive sum including nCluster+1 element  (nCluster+1)th element is the total number of found lines.
    //       checkGPUError(cudaMemcpyAsync(exclusiveFoundLinesHost.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceNExclusiveFoundLines(), (nClusters) * sizeof(int), cudaMemcpyDeviceToHost, mTimeFrameGPU->getStream(chunkId).get()));
    //       checkGPUError(cudaMemcpyAsync(&lastFoundLines, mTimeFrameGPU->getChunk(chunkId).getDeviceNFoundLines() + nClusters - 1, sizeof(int), cudaMemcpyDeviceToHost, mTimeFrameGPU->getStream(chunkId).get()));
    //       exclusiveFoundLinesHost[nClusters] = exclusiveFoundLinesHost[nClusters - 1] + lastFoundLines;

    //       std::vector<Line> lines(exclusiveFoundLinesHost[nClusters]);

    //       checkGPUError(cudaMemcpyAsync(lines.data(), mTimeFrameGPU->getChunk(chunkId).getDeviceLines(), sizeof(Line) * lines.size(), cudaMemcpyDeviceToHost, mTimeFrameGPU->getStream(chunkId).get()));
    //       checkGPUError(cudaStreamSynchronize(mTimeFrameGPU->getStream(chunkId).get()));

    //       // Compute vertices
    //       std::vector<ClusterLines> clusterLines;
    //       std::vector<bool> usedLines;
    //       for (int rofId{0}; rofId < rofs; ++rofId) {
    //         auto rof = offset + rofId;
    //         auto clustersL1offsetRof = mTimeFrameGPU->getROframeClusters(1)[rof] - mTimeFrameGPU->getROframeClusters(1)[offset]; // starting cluster offset for this ROF
    //         auto nClustersL1Rof = mTimeFrameGPU->getROframeClusters(1)[rof + 1] - mTimeFrameGPU->getROframeClusters(1)[rof];     // number of clusters for this ROF
    //         auto linesOffsetRof = exclusiveFoundLinesHost[clustersL1offsetRof];                                                  // starting line offset for this ROF
    //         auto nLinesRof = exclusiveFoundLinesHost[clustersL1offsetRof + nClustersL1Rof] - linesOffsetRof;
    //         gsl::span<const o2::its::Line> linesInRof(lines.data() + linesOffsetRof, static_cast<gsl::span<o2::its::Line>::size_type>(nLinesRof));

    //         usedLines.resize(linesInRof.size(), false);
    //         usedLines.assign(linesInRof.size(), false);
    //         clusterLines.clear();
    //         clusterLines.reserve(nClustersL1Rof);
    //         computeVerticesInRof(rof,
    //                              linesInRof,
    //                              usedLines,
    //                              clusterLines,
    //                              mTimeFrameGPU->getBeamXY(),
    //                              mTimeFrameGPU->getVerticesInChunks()[chunkId],
    //                              mTimeFrameGPU->getNVerticesInChunks()[chunkId],
    //                              mTimeFrameGPU,
    //                              mTimeFrameGPU->hasMCinformation() ? &mTimeFrameGPU->getLabelsInChunks()[chunkId] : nullptr);
    //       }
    //       offset += rofs;
    //     }
    //   };
    //   // Do work
    //   threads[chunkId] = std::thread(doVertexReconstruction);
    // }
    // for (auto& thread : threads) {
    //   thread.join();
    // }
    // for (int chunkId{0}; chunkId < mTimeFrameGPU->getNChunks(); ++chunkId) {
    //   int start{0};
    //   for (int rofId{0}; rofId < mTimeFrameGPU->getNVerticesInChunks()[chunkId].size(); ++rofId) {
    //     gsl::span<const Vertex> rofVerts{mTimeFrameGPU->getVerticesInChunks()[chunkId].data() + start, static_cast<gsl::span<Vertex>::size_type>(mTimeFrameGPU->getNVerticesInChunks()[chunkId][rofId])};
    //     mTimeFrameGPU->addPrimaryVertices(rofVerts);
    //     if (mTimeFrameGPU->hasMCinformation()) {
    //       mTimeFrameGPU->getVerticesLabels().emplace_back();
    //       // TODO: add MC labels
    //     }
    //     start += mTimeFrameGPU->getNVerticesInChunks()[chunkId][rofId];
    //   }
    // }
    // mTimeFrameGPU->wipe(3);
  }
}

void VertexerTraitsGPU::computeTrackletMatching(const int iteration)
{
}

void VertexerTraitsGPU::computeVertices(const int iteration)
{
}

void VertexerTraitsGPU::computeVerticesHist()
{
}

VertexerTraits* createVertexerTraitsGPU()
{
  return new VertexerTraitsGPU;
}
} // namespace o2::its
