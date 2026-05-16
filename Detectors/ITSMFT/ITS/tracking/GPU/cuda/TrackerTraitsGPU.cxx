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
#include "ITStracking/Configuration.h"

namespace o2::its
{

template <int NLayers>
void TrackerTraitsGPU<NLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialise(this->mTrkParams[iteration], NLayers, iteration);

  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
    // on default stream
    mTimeFrameGPU->loadVertices();
    // TODO these tables can be put in persistent memory
    mTimeFrameGPU->loadROFOverlapTable(); // this can be put in constant memory actually
    mTimeFrameGPU->loadROFVertexLookupTable();
    mTimeFrameGPU->loadTrackingTopologies();
    // once the tables are in persistent memory just update the vertex one
    // mTimeFrameGPU->updateROFVertexLookupTable();
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
  // push every create artefact on the stack
  mTimeFrameGPU->pushMemoryStack(iteration);
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
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NLayers; ++iLayer) {
    if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
      mTimeFrameGPU->createUsedClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersIndexTables(iLayer);
      mTimeFrameGPU->loadROFrameClustersDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int transitionId{0}; transitionId < hostTopology.nTransitions; ++transitionId) {
    const auto transition = hostTopology.getTransition(transitionId);
    mTimeFrameGPU->createTrackletsLUTDevice(this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass], transitionId);
    mTimeFrameGPU->waitEvent(transitionId, transition.fromLayer);
    mTimeFrameGPU->waitEvent(transitionId, transition.toLayer);
    countTrackletsInROFsHandler<NLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceROFMaskTableView(),
                                         transitionId,
                                         transition.fromLayer,
                                         transition.toLayer,
                                         mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                         mTimeFrameGPU->getDeviceROFVertexLookupTableView(),
                                         iVertex,
                                         mTimeFrameGPU->getDeviceVertices(),
                                         mTimeFrameGPU->getDeviceROFramesPV(),
                                         mTimeFrameGPU->getDeviceArrayClusters(),
                                         mTimeFrameGPU->getClusterSizes(),
                                         mTimeFrameGPU->getDeviceROFrameClusters(),
                                         (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                         mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                         mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                         mTimeFrameGPU->getDeviceTrackletsLUTs(),
                                         this->mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices],
                                         this->mTrkParams[iteration].NSigmaCut,
                                         topology,
                                         mTimeFrameGPU->getTransitionPhiCuts(),
                                         this->mTrkParams[iteration].PVres,
                                         mTimeFrameGPU->getMinRs(),
                                         mTimeFrameGPU->getMaxRs(),
                                         mTimeFrameGPU->getPositionResolutions(),
                                         this->mTrkParams[iteration].LayerRadii,
                                         mTimeFrameGPU->getTransitionMSAngles(),
                                         mTimeFrameGPU->getFrameworkAllocator(),
                                         mTimeFrameGPU->getStreams());
    mTimeFrameGPU->createTrackletsBuffers(transitionId);
    if (mTimeFrameGPU->getNTracklets()[transitionId] == 0) {
      mTimeFrameGPU->recordEvent(transitionId);
      continue;
    }
    computeTrackletsInROFsHandler<NLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                           mTimeFrameGPU->getDeviceROFMaskTableView(),
                                           transitionId,
                                           transition.fromLayer,
                                           transition.toLayer,
                                           mTimeFrameGPU->getDeviceROFOverlapTableView(),
                                           mTimeFrameGPU->getDeviceROFVertexLookupTableView(),
                                           iVertex,
                                           mTimeFrameGPU->getDeviceVertices(),
                                           mTimeFrameGPU->getDeviceROFramesPV(),
                                           mTimeFrameGPU->getDeviceArrayClusters(),
                                           mTimeFrameGPU->getClusterSizes(),
                                           mTimeFrameGPU->getDeviceROFrameClusters(),
                                           (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                           mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                           mTimeFrameGPU->getDeviceArrayTracklets(),
                                           mTimeFrameGPU->getDeviceTracklets(),
                                           mTimeFrameGPU->getNTracklets(),
                                           mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                           mTimeFrameGPU->getDeviceTrackletsLUTs(),
                                           this->mTrkParams[iteration].PassFlags[IterationStep::SelectUPCVertices],
                                           this->mTrkParams[iteration].NSigmaCut,
                                           topology,
                                           mTimeFrameGPU->getTransitionPhiCuts(),
                                           this->mTrkParams[iteration].PVres,
                                           mTimeFrameGPU->getMinRs(),
                                           mTimeFrameGPU->getMaxRs(),
                                           mTimeFrameGPU->getPositionResolutions(),
                                           this->mTrkParams[iteration].LayerRadii,
                                           mTimeFrameGPU->getTransitionMSAngles(),
                                           mTimeFrameGPU->getFrameworkAllocator(),
                                           mTimeFrameGPU->getStreams());
    mTimeFrameGPU->recordEvent(transitionId);
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
    const auto first = hostTopology.getTransition(cellTopology.firstTransition);
    const auto second = hostTopology.getTransition(cellTopology.secondTransition);
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[cellTopology.firstTransition])};
    if (!currentLayerTrackletsNum || !mTimeFrameGPU->getNTracklets()[cellTopology.secondTransition]) {
      mTimeFrameGPU->getNCells()[cellTopologyId] = 0;
      continue;
    }

    mTimeFrameGPU->createCellsLUTDevice(cellTopologyId);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.firstTransition);
    mTimeFrameGPU->waitEvent(cellTopologyId, cellTopology.secondTransition);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.fromLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, first.toLayer);
    mTimeFrameGPU->waitEvent(cellTopologyId, second.toLayer);
    countCellsHandler<NLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                               mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                               mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                               mTimeFrameGPU->getDeviceArrayTracklets(),
                               mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                               currentLayerTrackletsNum,
                               cellTopologyId,
                               topology,
                               nullptr,
                               mTimeFrameGPU->getDeviceArrayCellsLUT(),
                               mTimeFrameGPU->getDeviceCellLUTs()[cellTopologyId],
                               this->mBz,
                               this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                               this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                               this->mTrkParams[iteration].NSigmaCut,
                               this->mTrkParams[iteration].LayerxX0,
                               mTimeFrameGPU->getFrameworkAllocator(),
                               mTimeFrameGPU->getStreams());
    mTimeFrameGPU->createCellsBuffers(cellTopologyId);
    if (mTimeFrameGPU->getNCells()[cellTopologyId] == 0) {
      mTimeFrameGPU->recordEvent(cellTopologyId);
      continue;
    }
    computeCellsHandler<NLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                                 mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                 mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                 mTimeFrameGPU->getDeviceArrayTracklets(),
                                 mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                 currentLayerTrackletsNum,
                                 cellTopologyId,
                                 topology,
                                 mTimeFrameGPU->getDeviceCells()[cellTopologyId],
                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                 mTimeFrameGPU->getDeviceCellLUTs()[cellTopologyId],
                                 this->mBz,
                                 this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                 this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                                 this->mTrkParams[iteration].NSigmaCut,
                                 this->mTrkParams[iteration].LayerxX0,
                                 mTimeFrameGPU->getStreams());
    mTimeFrameGPU->recordEvent(cellTopologyId);
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  for (int outerLayer{0}; outerLayer < NLayers; ++outerLayer) {
    for (int targetCellTopologyId{0}; targetCellTopologyId < hostTopology.nCells; ++targetCellTopologyId) {
      const auto targetCellTopology = hostTopology.getCell(targetCellTopologyId);
      if (targetCellTopology.hitLayerMask.last() != outerLayer) {
        continue;
      }
      const int targetCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[targetCellTopologyId])};
      if (!targetCellsNum) {
        mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] = 0;
        mTimeFrameGPU->recordEvent(targetCellTopologyId);
        continue;
      }
      mTimeFrameGPU->createNeighboursIndexTablesDevice(targetCellTopologyId);
      mTimeFrameGPU->createNeighboursLUTDevice(targetCellTopologyId, targetCellsNum);

      for (int sourceCellTopologyId{0}; sourceCellTopologyId < hostTopology.nCells; ++sourceCellTopologyId) {
        const auto sourceCellTopology = hostTopology.getCell(sourceCellTopologyId);
        const int sourceCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[sourceCellTopologyId])};
        if (!sourceCellsNum || sourceCellTopology.secondTransition != targetCellTopology.firstTransition) {
          continue;
        }
        mTimeFrameGPU->waitEvent(targetCellTopologyId, sourceCellTopologyId);
        countCellNeighboursHandler<NLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                            mTimeFrameGPU->getDeviceNeighboursIndexTables(targetCellTopologyId),
                                            mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                            sourceCellTopologyId,
                                            targetCellTopologyId,
                                            this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                            this->mBz,
                                            sourceCellsNum,
                                            mTimeFrameGPU->getStream(targetCellTopologyId));
      }

      scanCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighboursIndexTables(targetCellTopologyId),
                                mTimeFrameGPU->getDeviceNeighboursLUT(targetCellTopologyId),
                                targetCellsNum,
                                mTimeFrameGPU->getFrameworkAllocator(),
                                mTimeFrameGPU->getStream(targetCellTopologyId));

      mTimeFrameGPU->createNeighboursDevice(targetCellTopologyId);
      if (mTimeFrameGPU->getNNeighbours()[targetCellTopologyId] == 0) {
        mTimeFrameGPU->recordEvent(targetCellTopologyId);
        continue;
      }

      for (int sourceCellTopologyId{0}; sourceCellTopologyId < hostTopology.nCells; ++sourceCellTopologyId) {
        const auto sourceCellTopology = hostTopology.getCell(sourceCellTopologyId);
        const int sourceCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[sourceCellTopologyId])};
        if (!sourceCellsNum || sourceCellTopology.secondTransition != targetCellTopology.firstTransition) {
          continue;
        }
        computeCellNeighboursHandler<NLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                              mTimeFrameGPU->getDeviceNeighboursIndexTables(targetCellTopologyId),
                                              mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                              mTimeFrameGPU->getDeviceNeighbours(targetCellTopologyId),
                                              sourceCellTopologyId,
                                              targetCellTopologyId,
                                              this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                              this->mBz,
                                              sourceCellsNum,
                                              mTimeFrameGPU->getStream(targetCellTopologyId));
      }
      mTimeFrameGPU->recordEvent(targetCellTopologyId);
    }
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findRoads(const int iteration)
{
  bounded_vector<bounded_vector<int>> firstClusters(this->mTrkParams[iteration].NLayers, bounded_vector<int>(this->getMemoryPool().get()), this->getMemoryPool().get());
  bounded_vector<bounded_vector<int>> sharedFirstClusters(this->mTrkParams[iteration].NLayers, bounded_vector<int>(this->getMemoryPool().get()), this->getMemoryPool().get());
  firstClusters.resize(this->mTrkParams[iteration].NLayers);
  sharedFirstClusters.resize(this->mTrkParams[iteration].NLayers);
  const auto hostTopology = mTimeFrameGPU->getTrackingTopologyView();
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    bounded_vector<TrackSeed<NLayers>> trackSeeds(this->getMemoryPool().get());
    for (int startCellTopologyId{0}; startCellTopologyId < hostTopology.nCells; ++startCellTopologyId) {
      const int startLayer = hostTopology.getCell(startCellTopologyId).hitLayerMask.last();
      if (!(this->mTrkParams[iteration].StartLayerMask.has(startLayer)) || mTimeFrameGPU->getNCells()[startCellTopologyId] == 0) {
        continue;
      }
      processNeighboursHandler<NLayers>(startLevel,
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
                                        trackSeeds,
                                        this->mBz,
                                        this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                        this->mTrkParams[iteration].MaxChi2NDF,
                                        this->mTrkParams[iteration].MaxHoles,
                                        this->mTrkParams[iteration].MinTrackLength,
                                        this->mTrkParams[iteration].HoleLayerMask,
                                        this->mTrkParams[iteration].LayerxX0,
                                        mTimeFrameGPU->getDevicePropagator(),
                                        this->mTrkParams[iteration].CorrType,
                                        mTimeFrameGPU->getFrameworkAllocator());
    }
    // fixme: I don't want to move tracks back and forth, but I need a way to use a thrust::allocator that is aware of our managed memory.
    if (trackSeeds.empty()) {
      LOGP(debug, "No track seeds found, skipping track finding");
      continue;
    }
    mTimeFrameGPU->loadTrackSeedsDevice(trackSeeds);

    // Since TrackITSExt is an enourmous class it is better to first count how many
    // successfull fits we do and only then allocate
    countTrackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),
                          mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                          mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                          mTimeFrameGPU->getDeviceTrackSeedsLUT(),
                          this->mTrkParams[iteration].LayerRadii,
                          this->mTrkParams[iteration].MinPt,
                          this->mTrkParams[iteration].LayerxX0,
                          trackSeeds.size(),
                          this->mBz,
                          startLevel,
                          this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                          this->mTrkParams[iteration].MaxChi2NDF,
                          this->mTrkParams[iteration].ReseedIfShorter,
                          this->mTrkParams[iteration].RepeatRefitOut,
                          this->mTrkParams[iteration].ShiftRefToCluster,
                          mTimeFrameGPU->getDevicePropagator(),
                          this->mTrkParams[iteration].CorrType,
                          mTimeFrameGPU->getFrameworkAllocator());
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds.size());
    computeTrackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),
                            mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                            mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                            mTimeFrameGPU->getDeviceTrackITSExt(),
                            mTimeFrameGPU->getDeviceTrackSeedsLUT(),
                            this->mTrkParams[iteration].LayerRadii,
                            this->mTrkParams[iteration].MinPt,
                            this->mTrkParams[iteration].LayerxX0,
                            trackSeeds.size(),
                            mTimeFrameGPU->getNTrackSeeds(),
                            this->mBz,
                            startLevel,
                            this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                            this->mTrkParams[iteration].MaxChi2NDF,
                            this->mTrkParams[iteration].ReseedIfShorter,
                            this->mTrkParams[iteration].RepeatRefitOut,
                            this->mTrkParams[iteration].ShiftRefToCluster,
                            mTimeFrameGPU->getDevicePropagator(),
                            this->mTrkParams[iteration].CorrType,
                            mTimeFrameGPU->getFrameworkAllocator());
    mTimeFrameGPU->downloadTrackITSExtDevice();

    auto& tracks = mTimeFrameGPU->getTrackITSExt();
    this->acceptTracks(iteration, tracks, firstClusters);
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
#endif
} // namespace o2::its
