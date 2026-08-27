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

#include <unistd.h>

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/LaunchGeometry.h"
#include "ITStracking/Configuration.h"

namespace o2::its
{

using o2::itsmft::tracking::CapacityEstimator;
using o2::itsmft::tracking::runOnSlab;
using o2::itsmft::tracking::SlabSite;

template <int NLayers>
void TrackerTraitsGPU<NLayers>::initialiseTimeFrame(const int iteration)
{
  this->mTaskArena->execute([&] {
    mTimeFrameGPU->initialise(this->mTrkParams[iteration], this->mTrkParams[iteration].NLayers, iteration);
  });
  // load iteration parameters
  mTimeFrameGPU->loadIterationParameters(this->mTrkParams[iteration]);

  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
    // on default stream
    mTimeFrameGPU->loadVertices();
    // TODO these tables can be put in persistent memory
    mTimeFrameGPU->loadROFOverlapTable(); // this can be put in constant memory actually
    mTimeFrameGPU->loadROFVertexLookupTable();
    mTimeFrameGPU->loadTrackingTopologies();
    // once the tables are in persistent memory just re-upload the vertex one
    // mTimeFrameGPU->uploadROFVertexLookupTable();
    mTimeFrameGPU->loadIndexTableUtils();
    // pinned on host
    mTimeFrameGPU->createUsedClustersDeviceArray();
    mTimeFrameGPU->createClustersDeviceArray();
    mTimeFrameGPU->createUnsortedClustersDeviceArray();
    mTimeFrameGPU->createClustersIndexTablesArray();
    mTimeFrameGPU->createTrackingFrameInfoDeviceArray();
    mTimeFrameGPU->createROFrameClustersDeviceArray();
    // device array
    mTimeFrameGPU->createTrackletsLUTDeviceArray();
    mTimeFrameGPU->createTrackletsBuffersArray();
    mTimeFrameGPU->createCellsBuffersArray();
    mTimeFrameGPU->createCellsLUTDeviceArray();
  }
  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass] || this->mTrkParams[iteration].PassFlags[IterationStep::UseUPCMask]) {
    mTimeFrameGPU->loadROFCutMask(iteration);
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::adoptTimeFrame(TimeFrame<NLayers>* tf)
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<NLayers>*>(tf);
  this->mTimeFrame = static_cast<TimeFrame<NLayers>*>(tf);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerTracklets(const int iteration, int iVertex)
{
  const auto topology = mTimeFrameGPU->getDeviceTrackingTopologyView();
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  const bool loadFirstPassData = this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass] && iVertex <= 0; // load data only on first pass and first vertex
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NLayers; ++iLayer) {
    if (loadFirstPassData) {
      mTimeFrameGPU->createUsedClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersIndexTables(iLayer);
      mTimeFrameGPU->loadROFrameClustersDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int linkId{0}; linkId < hostTopology.nLinks; ++linkId) {
    mTimeFrameGPU->createTrackletsLUTDevice(loadFirstPassData, linkId); // on first pass allocates, then only clears memory
  }

  // Stack allocations created from trackleting through road finding are scoped to one tracker pass.
  // With per-primary-vertex processing, the chain is called once per vertex while initialisation is only done once.
  mTimeFrameGPU->pushMemoryStack(iteration);

  const auto nClusters = mTimeFrameGPU->getClusterSizes();
  for (int linkId{0}; linkId < hostTopology.nLinks; ++linkId) {
    const auto link = hostTopology.getLink(linkId);
    mTimeFrameGPU->waitEvent(linkId, link.fromLayer);
    mTimeFrameGPU->waitEvent(linkId, link.toLayer);
    const auto key = CapacityEstimator::makeKey(SlabSite::Tracklets, iteration, iVertex + 1, linkId);
    const auto scale = static_cast<double>(nClusters[link.fromLayer]);
    runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
      mTimeFrameGPU->createTrackletsBuffers(linkId, capacity);
      return TrackingKernels<NLayers>::computeTrackletsInROFsHandler(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                                                     mTimeFrameGPU->getDeviceROFMaskTableView(),
                                                                     linkId,
                                                                     link.fromLayer,
                                                                     link.toLayer,
                                                                     mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                                                     mTimeFrameGPU->getDeviceROFVertexLookupTableView(),
                                                                     iVertex,
                                                                     mTimeFrameGPU->getDeviceVertices(),
                                                                     mTimeFrameGPU->getDeviceArrayClusters(),
                                                                     nClusters,
                                                                     mTimeFrameGPU->getDeviceROFrameClusters(),
                                                                     (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                                     mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                                                     mTimeFrameGPU->getDeviceArrayTracklets(),
                                                                     mTimeFrameGPU->getDeviceTracklets(),
                                                                     mTimeFrameGPU->getNTracklets(),
                                                                     capacity,
                                                                     mTimeFrameGPU->getDeviceTrackletsLUTs(),
                                                                     this->mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices],
                                                                     this->mTrkParams[iteration].NSigmaCut,
                                                                     topology,
                                                                     mTimeFrameGPU->getLinkPhiCuts(),
                                                                     this->mTrkParams[iteration].PVres,
                                                                     mTimeFrameGPU->getMinRs(),
                                                                     mTimeFrameGPU->getMaxRs(),
                                                                     mTimeFrameGPU->getPositionResolutions(),
                                                                     this->mTrkParams[iteration].LayerRadii,
                                                                     mTimeFrameGPU->getLinkMSAngles(),
                                                                     mTimeFrameGPU->getFrameworkAllocator(),
                                                                     mTimeFrameGPU->getStreams());
    });
    mTimeFrameGPU->recordEvent(linkId);
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells(const int iteration)
{
  const auto topology = mTimeFrameGPU->getDeviceTrackingTopologyView();
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NLayers; ++iLayer) {
    if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
      mTimeFrameGPU->loadUnsortedClustersDevice(iLayer);
      mTimeFrameGPU->loadTrackingFrameInfoDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int cellTopologyId{hostTopology.nCells}; cellTopologyId--;) {
    const auto cellTopology = hostTopology.getCell(cellTopologyId);
    const auto first = hostTopology.getLink(cellTopology.firstLink);
    const auto second = hostTopology.getLink(cellTopology.secondLink);
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[cellTopology.firstLink])};
    if (!currentLayerTrackletsNum || !mTimeFrameGPU->getNTracklets()[cellTopology.secondLink]) {
      mTimeFrameGPU->getNCells()[cellTopologyId] = 0;
      continue;
    }

    mTimeFrameGPU->createCellsLUTDevice(cellTopologyId);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.firstLink);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.secondLink);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.fromLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.toLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, second.toLayer);
    const auto key = CapacityEstimator::makeKey(SlabSite::Cells, iteration, 0, cellTopologyId);
    const auto scale = static_cast<double>(currentLayerTrackletsNum);
    const int emitted = runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
      mTimeFrameGPU->createCellsBuffers(cellTopologyId, capacity);
      return TrackingKernels<NLayers>::computeCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                                                           mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                                           mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                           mTimeFrameGPU->getDeviceArrayTracklets(),
                                                           mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                                           currentLayerTrackletsNum,
                                                           cellTopologyId,
                                                           topology,
                                                           mTimeFrameGPU->getDeviceCells()[cellTopologyId],
                                                           capacity,
                                                           mTimeFrameGPU->getDeviceCellLUTs()[cellTopologyId],
                                                           this->mBz,
                                                           this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                           this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                                                           this->mTrkParams[iteration].NSigmaCut,
                                                           mTimeFrameGPU->getDeviceLayerxX0(),
                                                           mTimeFrameGPU->getFrameworkAllocator(),
                                                           mTimeFrameGPU->getStreams());
    });
    mTimeFrameGPU->getNCells()[cellTopologyId] = emitted;
    mTimeFrameGPU->recordEvent(cellTopologyId);
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  bounded_vector<int> sourceTopologies(this->getMemoryPool().get());
  sourceTopologies.reserve(hostTopology.nCells);
  for (int outerLayer{0}; outerLayer < NLayers; ++outerLayer) {
    for (int targetCellTopologyId{0}; targetCellTopologyId < hostTopology.nCells; ++targetCellTopologyId) {
      const auto targetCellTopology = hostTopology.getCell(targetCellTopologyId);
      if (targetCellTopology.hitLayerMask.last() != outerLayer) {
        continue;
      }
      const int targetCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[targetCellTopologyId])};
      sourceTopologies.clear();
      size_t sourceCellCount{0};
      for (int sourceCellTopologyId{0}; sourceCellTopologyId < hostTopology.nCells; ++sourceCellTopologyId) {
        const auto sourceCellTopology = hostTopology.getCell(sourceCellTopologyId);
        const int sourceCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[sourceCellTopologyId])};
        if (!sourceCellsNum || sourceCellTopology.secondLink != targetCellTopology.firstLink) {
          continue;
        }
        sourceTopologies.push_back(sourceCellTopologyId);
        sourceCellCount += sourceCellsNum;
      }
      if (!targetCellsNum || sourceTopologies.empty()) {
        mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] = 0;
        mTimeFrameGPU->createNeighboursDevice(targetCellTopologyId, 0);
        mTimeFrameGPU->recordEvent(targetCellTopologyId);
        continue;
      }
      mTimeFrameGPU->createNeighboursLUTDevice(targetCellTopologyId, targetCellsNum);
      auto& stream = mTimeFrameGPU->getStream(targetCellTopologyId);
      int* outputCounter = mTimeFrameGPU->getDeviceNeighboursLUT(targetCellTopologyId) + targetCellsNum;

      const auto key = CapacityEstimator::makeKey(SlabSite::Neighbours, iteration, 0, targetCellTopologyId);
      const auto scale = static_cast<double>(sourceCellCount);
      const int emitted = runOnSlab(mTimeFrameGPU->getCapacityEstimator(), key, scale, [&](const int capacity) {
        mTimeFrameGPU->createNeighboursDevice(targetCellTopologyId, capacity);
        resetOutputCounterHandler(outputCounter, stream);
        for (const int sourceCellTopologyId : sourceTopologies) {
          mTimeFrameGPU->waitEvent(targetCellTopologyId, sourceCellTopologyId);
          TrackingKernels<NLayers>::computeCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                                                 mTimeFrameGPU->getDeviceNeighbours(targetCellTopologyId),
                                                                 outputCounter,
                                                                 capacity,
                                                                 sourceCellTopologyId,
                                                                 targetCellTopologyId,
                                                                 this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                                 this->mBz,
                                                                 mTimeFrameGPU->getNCells()[sourceCellTopologyId],
                                                                 mTimeFrameGPU->getFrameworkAllocator(),
                                                                 stream);
        }
        return finalizeCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighbours(targetCellTopologyId),
                                             mTimeFrameGPU->getDeviceNeighboursLUT(targetCellTopologyId),
                                             targetCellsNum,
                                             capacity,
                                             mTimeFrameGPU->getFrameworkAllocator(),
                                             stream);
      });
      mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] = emitted;
      mTimeFrameGPU->recordEvent(targetCellTopologyId);
    }
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(this->mTrkParams[iteration].NLayers, bounded_vector<int>(this->getMemoryPool().get()), this->getMemoryPool().get());
  firstClusters.resize(this->mTrkParams[iteration].NLayers);
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  const bool extendTop = this->mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerTop];
  const bool extendBot = this->mTrkParams[iteration].PassFlags[IterationStep::TrackFollowerBot];
  const bool extendTracks = extendTop || extendBot;
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    // The cells that may start a road at this level, as the scale the estimator predicts from.
    size_t startCells{0};
    for (int startCellTopologyId{0}; startCellTopologyId < hostTopology.nCells; ++startCellTopologyId) {
      const int startLayer = hostTopology.getCell(startCellTopologyId).hitLayerMask.last();
      if (this->mTrkParams[iteration].StartLayerMask.has(startLayer)) {
        startCells += mTimeFrameGPU->getNCells()[startCellTopologyId];
      }
    }
    if (!startCells) {
      continue;
    }
    const auto key = CapacityEstimator::makeKey(SlabSite::TrackSeeds, iteration, startLevel, 0);
    auto& estimator = mTimeFrameGPU->getCapacityEstimator();
    const int nSeeds = runOnSlab(estimator, key, static_cast<double>(startCells), [&](const int capacity) {
      mTimeFrameGPU->createTrackSeedsDevice(capacity);
      int cursor{0};
      for (int startCellTopologyId{0}; startCellTopologyId < hostTopology.nCells; ++startCellTopologyId) {
        const int startLayer = hostTopology.getCell(startCellTopologyId).hitLayerMask.last();
        if (!(this->mTrkParams[iteration].StartLayerMask.has(startLayer)) || mTimeFrameGPU->getNCells()[startCellTopologyId] == 0) {
          continue;
        }
        TrackingKernels<NLayers>::processNeighboursHandler(startLevel,
                                                           startCellTopologyId,
                                                           mTimeFrameGPU->getDeviceArrayCells(),
                                                           mTimeFrameGPU->getDeviceCells()[startCellTopologyId],
                                                           nullptr,
                                                           nullptr,
                                                           mTimeFrameGPU->getArrayNCells().data(),
                                                           (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                           mTimeFrameGPU->getDeviceArrayNeighbours(),
                                                           mTimeFrameGPU->getDeviceArrayNeighboursCellLUT(),
                                                           mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                           mTimeFrameGPU->getDeviceTrackSeeds(),
                                                           capacity,
                                                           cursor,
                                                           mTimeFrameGPU->getCapacityEstimator(),
                                                           iteration,
                                                           this->mBz,
                                                           this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                           this->mTrkParams[iteration].MaxChi2NDF,
                                                           this->mTrkParams[iteration].MaxHoles,
                                                           this->mTrkParams[iteration].getMinSeedingClusters(),
                                                           this->mTrkParams[iteration].HoleLayerMask,
                                                           this->mTrkParams[iteration].getNonSeedingLayerMask(),
                                                           mTimeFrameGPU->getDeviceLayerxX0(),
                                                           mTimeFrameGPU->getDevicePropagator(),
                                                           this->mTrkParams[iteration].CorrType,
                                                           mTimeFrameGPU->getFrameworkAllocator());
      }
      return cursor; }, estimator.peakCapacity(key));
    if (!nSeeds) {
      LOGP(debug, "No track seeds found, skipping track finding");
      continue;
    }
    if (extendTracks) { // independent of the slab size, so it must not be redone on a retry
      mTimeFrameGPU->createTrackExtensionScratchDevice(gpu::gridThreads(gpu::ResidentBlocks.fitTrackSeedsExtended),
                                                       this->mTrkParams[iteration].TrackFollowerMaxHypotheses);
    }
    const auto trackKey = CapacityEstimator::makeKey(extendTracks ? SlabSite::TracksExtended : SlabSite::Tracks,
                                                     iteration, startLevel, 0);
    const int nTracks = runOnSlab(estimator, trackKey, static_cast<double>(nSeeds), [&](const int capacity) {
      mTimeFrameGPU->createTrackITSExtDevice(capacity);
      return TrackingKernels<NLayers>::computeTrackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),
                                                               mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                                               mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                                               mTimeFrameGPU->getDeviceIndexTableUtils(),
                                                               mTimeFrameGPU->getDeviceROFMaskTableView(),
                                                               mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                                               mTimeFrameGPU->getDeviceArrayClusters(),
                                                               (const unsigned char**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                                               mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                                               mTimeFrameGPU->getDeviceROFrameClusters(),
                                                               mTimeFrameGPU->getDeviceTrackITSExt(),
                                                               mTimeFrameGPU->getDeviceTrackIndices(),
                                                               mTimeFrameGPU->getDeviceTrackSeedIndices(),
                                                               mTimeFrameGPU->getDeviceTrackCounter(),
                                                               capacity,
                                                               extendTracks ? mTimeFrameGPU->getDeviceActiveTrackExtensionHypotheses() : nullptr,
                                                               extendTracks ? mTimeFrameGPU->getDeviceNextTrackExtensionHypotheses() : nullptr,
                                                               mTimeFrameGPU->getDeviceLayerRadii(),
                                                               mTimeFrameGPU->getDeviceMinPts(),
                                                               mTimeFrameGPU->getDeviceLayerxX0(),
                                                               static_cast<unsigned int>(nSeeds),
                                                               this->mBz,
                                                               this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                                               this->mTrkParams[iteration].MaxChi2NDF,
                                                               this->mTrkParams[iteration].ReseedIfShorter,
                                                               this->mTrkParams[iteration].RepeatRefitOut,
                                                               this->mTrkParams[iteration].ShiftRefToCluster,
                                                               this->mTrkParams[iteration].NLayers,
                                                               this->mTrkParams[iteration].PhiBins,
                                                               this->mTrkParams[iteration].TrackFollowerMaxHypotheses,
                                                               extendTop,
                                                               extendBot,
                                                               this->mTrkParams[iteration].TrackFollowerNSigmaCutPhi,
                                                               this->mTrkParams[iteration].TrackFollowerNSigmaCutZ,
                                                               mTimeFrameGPU->getDevicePropagator(),
                                                               this->mTrkParams[iteration].CorrType,
                                                               mTimeFrameGPU->getFrameworkAllocator()); }, estimator.peakCapacity(trackKey));
    mTimeFrameGPU->createTrackITSExtHost(nTracks);
    mTimeFrameGPU->downloadTrackITSExtDevice();

    auto& tracks = mTimeFrameGPU->getTrackITSExt();
    const auto& trackIndices = mTimeFrameGPU->getTrackIndices();
    this->acceptTracks(iteration, tracks, trackIndices, firstClusters);
    mTimeFrameGPU->loadUsedClustersDevice();
  }
  this->markTracks(iteration);
  // wipe the artefact memory
  mTimeFrameGPU->popMemoryStack(iteration);
};

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfTracklets() const
{
  return std::accumulate(mTimeFrameGPU->getNTracklets().begin(), mTimeFrameGPU->getNTracklets().end(), 0);
}

template <int NLayers>
int TrackerTraitsGPU<NLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::setBz(float bz)
{
  this->mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template class TrackerTraitsGPU<7>;
#ifdef ENABLE_UPGRADES
template class TrackerTraitsGPU<11>;
template class TrackerTraitsGPU<13>;
#endif
} // namespace o2::its
