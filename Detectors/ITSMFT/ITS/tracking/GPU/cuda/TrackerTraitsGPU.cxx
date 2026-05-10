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
  mTimeFrameGPU->initialise(this->mTrkParams[iteration], NLayers);

  if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
    // on default stream
    mTimeFrameGPU->loadVertices();
    // TODO these tables can be put in persistent memory
    mTimeFrameGPU->loadROFOverlapTable(); // this can be put in constant memory actually
    mTimeFrameGPU->loadROFVertexLookupTable();
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
  // start by queuing loading needed of two last layers
  for (int iLayer{NLayers}; iLayer-- > NLayers - 2;) {
    if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
      mTimeFrameGPU->createUsedClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersDevice(iLayer);
      mTimeFrameGPU->loadClustersIndexTables(iLayer);
      mTimeFrameGPU->loadROFrameClustersDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int iLayer{this->mTrkParams[iteration].TrackletsPerRoad()}; iLayer--;) {
    if (iLayer) { // queue loading data of next layer in parallel, this the copies are overlapping with computation kernels
      if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
        mTimeFrameGPU->createUsedClustersDevice(iLayer - 1);
        mTimeFrameGPU->loadClustersDevice(iLayer - 1);
        mTimeFrameGPU->loadClustersIndexTables(iLayer - 1);
        mTimeFrameGPU->loadROFrameClustersDevice(iLayer - 1);
      }
      mTimeFrameGPU->recordEvent(iLayer - 1);
    }
    mTimeFrameGPU->createTrackletsLUTDevice(this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass], iLayer);
    mTimeFrameGPU->waitEvent(iLayer, iLayer + 1); // wait stream until all data is available
    countTrackletsInROFsHandler<NLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceROFMaskTableView(),
                                         iLayer,
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
                                         mTimeFrameGPU->getPhiCuts(),
                                         this->mTrkParams[iteration].PVres,
                                         mTimeFrameGPU->getMinRs(),
                                         mTimeFrameGPU->getMaxRs(),
                                         mTimeFrameGPU->getPositionResolutions(),
                                         this->mTrkParams[iteration].LayerRadii,
                                         mTimeFrameGPU->getMSangles(),
                                         mTimeFrameGPU->getFrameworkAllocator(),
                                         mTimeFrameGPU->getStreams());
    mTimeFrameGPU->createTrackletsBuffers(iLayer);
    if (mTimeFrameGPU->getNTracklets()[iLayer] == 0) {
      continue;
    }
    computeTrackletsInROFsHandler<NLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                           mTimeFrameGPU->getDeviceROFMaskTableView(),
                                           iLayer,
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
                                           mTimeFrameGPU->getPhiCuts(),
                                           this->mTrkParams[iteration].PVres,
                                           mTimeFrameGPU->getMinRs(),
                                           mTimeFrameGPU->getMaxRs(),
                                           mTimeFrameGPU->getPositionResolutions(),
                                           this->mTrkParams[iteration].LayerRadii,
                                           mTimeFrameGPU->getMSangles(),
                                           mTimeFrameGPU->getFrameworkAllocator(),
                                           mTimeFrameGPU->getStreams());
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells(const int iteration)
{
  // start by queuing loading needed of three last layers
  for (int iLayer{NLayers}; iLayer-- > NLayers - 3;) {
    if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
      mTimeFrameGPU->loadUnsortedClustersDevice(iLayer);
      mTimeFrameGPU->loadTrackingFrameInfoDevice(iLayer);
    }
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int iLayer{this->mTrkParams[iteration].CellsPerRoad()}; iLayer--;) {
    if (iLayer) {
      if (this->mTrkParams[iteration].PassFlags[IterationStep::FirstPass]) {
        mTimeFrameGPU->loadUnsortedClustersDevice(iLayer - 1);
        mTimeFrameGPU->loadTrackingFrameInfoDevice(iLayer - 1);
      }
      mTimeFrameGPU->recordEvent(iLayer - 1);
    }

    // if there are no tracklets skip entirely
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[iLayer])};
    if (!currentLayerTrackletsNum || !mTimeFrameGPU->getNTracklets()[iLayer + 1]) {
      mTimeFrameGPU->getNCells()[iLayer] = 0;
      continue;
    }

    mTimeFrameGPU->createCellsLUTDevice(iLayer);
    mTimeFrameGPU->waitEvent(iLayer, iLayer + 1); // wait stream until all data is available
    mTimeFrameGPU->waitEvent(iLayer, iLayer + 2); // wait stream until all data is available
    countCellsHandler<NLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                               mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                               mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                               mTimeFrameGPU->getDeviceArrayTracklets(),
                               mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                               currentLayerTrackletsNum,
                               iLayer,
                               nullptr,
                               mTimeFrameGPU->getDeviceArrayCellsLUT(),
                               mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                               this->mBz,
                               this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                               this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                               this->mTrkParams[iteration].NSigmaCut,
                               this->mTrkParams[iteration].LayerxX0,
                               mTimeFrameGPU->getFrameworkAllocator(),
                               mTimeFrameGPU->getStreams());
    mTimeFrameGPU->createCellsBuffers(iLayer);
    if (mTimeFrameGPU->getNCells()[iLayer] == 0) {
      continue;
    }
    computeCellsHandler<NLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                                 mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                 mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                 mTimeFrameGPU->getDeviceArrayTracklets(),
                                 mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                 currentLayerTrackletsNum,
                                 iLayer,
                                 mTimeFrameGPU->getDeviceCells()[iLayer],
                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                 mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                                 this->mBz,
                                 this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                 this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                                 this->mTrkParams[iteration].NSigmaCut,
                                 this->mTrkParams[iteration].LayerxX0,
                                 mTimeFrameGPU->getStreams());
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findCellsNeighbours(const int iteration)
{
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NeighboursPerRoad(); ++iLayer) {
    if (iLayer > 0) {
      // Previous layer updates levels in this layer's cells.
      mTimeFrameGPU->waitEvent(iLayer, iLayer - 1);
    }
    const int currentLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer])};
    const int nextLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer + 1])};
    if (!nextLayerCellsNum || !currentLayerCellsNum) {
      mTimeFrameGPU->getNNeighbours()[iLayer] = 0;
      mTimeFrameGPU->recordEvent(iLayer);
      continue;
    }
    mTimeFrameGPU->createNeighboursIndexTablesDevice(iLayer);
    mTimeFrameGPU->createNeighboursLUTDevice(iLayer, nextLayerCellsNum);
    countCellNeighboursHandler<NLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                        mTimeFrameGPU->getDeviceNeighboursLUT(iLayer), // LUT is initialised here.
                                        mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                        mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                        mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                        (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                        this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                        this->mBz,
                                        iLayer,
                                        currentLayerCellsNum,
                                        nextLayerCellsNum,
                                        1e2,
                                        mTimeFrameGPU->getFrameworkAllocator(),
                                        mTimeFrameGPU->getStream(iLayer));
    mTimeFrameGPU->createNeighboursDevice(iLayer);
    if (mTimeFrameGPU->getNNeighbours()[iLayer] == 0) {
      mTimeFrameGPU->recordEvent(iLayer);
      continue;
    }
    computeCellNeighboursHandler<NLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                          mTimeFrameGPU->getDeviceNeighboursLUT(iLayer),
                                          mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                          mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                          mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                          (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                          this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                          this->mBz,
                                          iLayer,
                                          currentLayerCellsNum,
                                          nextLayerCellsNum,
                                          1e2,
                                          mTimeFrameGPU->getStream(iLayer));
    mTimeFrameGPU->getArrayNNeighbours()[iLayer] = filterCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                                                               mTimeFrameGPU->getDeviceNeighbours(iLayer),
                                                                               mTimeFrameGPU->getArrayNNeighbours()[iLayer],
                                                                               mTimeFrameGPU->getStream(iLayer),
                                                                               mTimeFrameGPU->getFrameworkAllocator());
    mTimeFrameGPU->recordEvent(iLayer);
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
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    bounded_vector<TrackSeed<NLayers>> trackSeeds(this->getMemoryPool().get());
    for (int startLayer{this->mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      if ((this->mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }
      processNeighboursHandler<NLayers>(startLayer,
                                        startLevel,
                                        mTimeFrameGPU->getDeviceArrayCells(),
                                        mTimeFrameGPU->getDeviceCells()[startLayer],
                                        mTimeFrameGPU->getArrayNCells(),
                                        (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                        mTimeFrameGPU->getDeviceNeighboursAll(),
                                        mTimeFrameGPU->getDeviceNeighboursLUTs(),
                                        mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                        trackSeeds,
                                        this->mBz,
                                        this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                        this->mTrkParams[iteration].MaxChi2NDF,
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
    this->acceptTracks(iteration, tracks, firstClusters, sharedFirstClusters);
    mTimeFrameGPU->loadUsedClustersDevice();
  }
  this->markTracks(iteration, sharedFirstClusters);
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
