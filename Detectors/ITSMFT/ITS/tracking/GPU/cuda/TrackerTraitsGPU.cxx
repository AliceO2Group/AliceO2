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
#include <vector>
#include <unistd.h>

#include "DataFormatsITS/TrackITS.h"

#include "ITStrackingGPU/TrackerTraitsGPU.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Constants.h"

namespace o2::its
{

template <int NLayers>
void TrackerTraitsGPU<NLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialise(iteration, this->mTrkParams[iteration], NLayers);
  // on default stream
  mTimeFrameGPU->loadVertices(iteration);
  // TODO these tables can be put in persistent memory
  mTimeFrameGPU->loadROFOverlapTable(iteration); // this can be put in constant memory actually
  mTimeFrameGPU->loadROFVertexLookupTable(iteration);
  // once the tables are in persistent memory just update the vertex one
  // mTimeFrameGPU->updateROFVertexLookupTable(iteration);
  mTimeFrameGPU->loadIndexTableUtils(iteration);
  mTimeFrameGPU->loadROFCutMask(iteration);
  // pinned on host
  mTimeFrameGPU->createUsedClustersDeviceArray(iteration);
  mTimeFrameGPU->createClustersDeviceArray(iteration);
  mTimeFrameGPU->createUnsortedClustersDeviceArray(iteration);
  mTimeFrameGPU->createClustersIndexTablesArray(iteration);
  mTimeFrameGPU->createTrackingFrameInfoDeviceArray(iteration);
  mTimeFrameGPU->createROFrameClustersDeviceArray(iteration);
  // device array
  mTimeFrameGPU->createTrackletsLUTDeviceArray(iteration);
  mTimeFrameGPU->createTrackletsBuffersArray(iteration);
  mTimeFrameGPU->createCellsBuffersArray(iteration);
  mTimeFrameGPU->createCellsLUTDeviceArray(iteration);
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
  const auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  // start by queuing loading needed of two last layers
  for (int iLayer{NLayers}; iLayer-- > NLayers - 2;) {
    mTimeFrameGPU->createUsedClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadClustersIndexTables(iteration, iLayer);
    mTimeFrameGPU->loadROFrameClustersDevice(iteration, iLayer);
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int iLayer{this->mTrkParams[iteration].TrackletsPerRoad()}; iLayer--;) {
    if (iLayer) { // queue loading data of next layer in parallel, this the copies are overlapping with computation kernels
      mTimeFrameGPU->createUsedClustersDevice(iteration, iLayer - 1);
      mTimeFrameGPU->loadClustersDevice(iteration, iLayer - 1);
      mTimeFrameGPU->loadClustersIndexTables(iteration, iLayer - 1);
      mTimeFrameGPU->loadROFrameClustersDevice(iteration, iLayer - 1);
      mTimeFrameGPU->recordEvent(iLayer - 1);
    }
    mTimeFrameGPU->createTrackletsLUTDevice(iteration, iLayer);
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
                                         iteration,
                                         this->mTrkParams[iteration].NSigmaCut,
                                         mTimeFrameGPU->getPhiCuts(),
                                         this->mTrkParams[iteration].PVres,
                                         mTimeFrameGPU->getMinRs(),
                                         mTimeFrameGPU->getMaxRs(),
                                         mTimeFrameGPU->getPositionResolutions(),
                                         this->mTrkParams[iteration].LayerRadii,
                                         mTimeFrameGPU->getMSangles(),
                                         mTimeFrameGPU->getFrameworkAllocator(),
                                         conf.nBlocksLayerTracklets[iteration],
                                         conf.nThreadsLayerTracklets[iteration],
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
                                           iteration,
                                           this->mTrkParams[iteration].NSigmaCut,
                                           mTimeFrameGPU->getPhiCuts(),
                                           this->mTrkParams[iteration].PVres,
                                           mTimeFrameGPU->getMinRs(),
                                           mTimeFrameGPU->getMaxRs(),
                                           mTimeFrameGPU->getPositionResolutions(),
                                           this->mTrkParams[iteration].LayerRadii,
                                           mTimeFrameGPU->getMSangles(),
                                           mTimeFrameGPU->getFrameworkAllocator(),
                                           conf.nBlocksLayerTracklets[iteration],
                                           conf.nThreadsLayerTracklets[iteration],
                                           mTimeFrameGPU->getStreams());
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::computeLayerCells(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  // start by queuing loading needed of three last layers
  for (int iLayer{NLayers}; iLayer-- > NLayers - 3;) {
    mTimeFrameGPU->loadUnsortedClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadTrackingFrameInfoDevice(iteration, iLayer);
    mTimeFrameGPU->recordEvent(iLayer);
  }

  for (int iLayer{this->mTrkParams[iteration].CellsPerRoad()}; iLayer--;) {
    if (iLayer) {
      mTimeFrameGPU->loadUnsortedClustersDevice(iteration, iLayer - 1);
      mTimeFrameGPU->loadTrackingFrameInfoDevice(iteration, iLayer - 1);
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
                               mTimeFrameGPU->getFrameworkAllocator(),
                               conf.nBlocksLayerCells[iteration],
                               conf.nThreadsLayerCells[iteration],
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
                                 conf.nBlocksLayerCells[iteration],
                                 conf.nThreadsLayerCells[iteration],
                                 mTimeFrameGPU->getStreams());
  }
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findCellsNeighbours(const int iteration)
{
  const auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].NeighboursPerRoad(); ++iLayer) {
    const int currentLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer])};
    const int nextLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer + 1])};
    if (!nextLayerCellsNum || !currentLayerCellsNum) {
      mTimeFrameGPU->getNNeighbours()[iLayer] = 0;
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
                                        conf.nBlocksFindNeighbours[iteration],
                                        conf.nThreadsFindNeighbours[iteration],
                                        mTimeFrameGPU->getStream(iLayer));
    mTimeFrameGPU->createNeighboursDevice(iLayer);
    if (mTimeFrameGPU->getNNeighbours()[iLayer] == 0) {
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
                                          conf.nBlocksFindNeighbours[iteration],
                                          conf.nThreadsFindNeighbours[iteration],
                                          mTimeFrameGPU->getStream(iLayer));
    mTimeFrameGPU->getArrayNNeighbours()[iLayer] = filterCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                                                               mTimeFrameGPU->getDeviceNeighbours(iLayer),
                                                                               mTimeFrameGPU->getArrayNNeighbours()[iLayer],
                                                                               mTimeFrameGPU->getStream(iLayer),
                                                                               mTimeFrameGPU->getFrameworkAllocator());
  }
  mTimeFrameGPU->syncStreams(false);
}

template <int NLayers>
void TrackerTraitsGPU<NLayers>::findRoads(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    bounded_vector<CellSeed<NLayers>> trackSeeds(this->getMemoryPool().get());
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
                                        this->mTrkParams[0].MaxChi2ClusterAttachment,
                                        this->mTrkParams[0].MaxChi2NDF,
                                        mTimeFrameGPU->getDevicePropagator(),
                                        this->mTrkParams[0].CorrType,
                                        mTimeFrameGPU->getFrameworkAllocator(),
                                        conf.nBlocksProcessNeighbours[iteration],
                                        conf.nThreadsProcessNeighbours[iteration]);
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
                          trackSeeds.size(),
                          this->mBz,
                          startLevel,
                          this->mTrkParams[0].MaxChi2ClusterAttachment,
                          this->mTrkParams[0].MaxChi2NDF,
                          this->mTrkParams[0].ReseedIfShorter,
                          this->mTrkParams[0].RepeatRefitOut,
                          this->mTrkParams[0].ShiftRefToCluster,
                          mTimeFrameGPU->getDevicePropagator(),
                          this->mTrkParams[0].CorrType,
                          mTimeFrameGPU->getFrameworkAllocator(),
                          conf.nBlocksTracksSeeds[iteration],
                          conf.nThreadsTracksSeeds[iteration]);
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds.size());
    computeTrackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),
                            mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                            mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                            mTimeFrameGPU->getDeviceTrackITSExt(),
                            mTimeFrameGPU->getDeviceTrackSeedsLUT(),
                            this->mTrkParams[iteration].LayerRadii,
                            this->mTrkParams[iteration].MinPt,
                            trackSeeds.size(),
                            mTimeFrameGPU->getNTrackSeeds(),
                            this->mBz,
                            startLevel,
                            this->mTrkParams[0].MaxChi2ClusterAttachment,
                            this->mTrkParams[0].MaxChi2NDF,
                            this->mTrkParams[0].ReseedIfShorter,
                            this->mTrkParams[0].RepeatRefitOut,
                            this->mTrkParams[0].ShiftRefToCluster,
                            mTimeFrameGPU->getDevicePropagator(),
                            this->mTrkParams[0].CorrType,
                            mTimeFrameGPU->getFrameworkAllocator(),
                            conf.nBlocksTracksSeeds[iteration],
                            conf.nThreadsTracksSeeds[iteration]);
    mTimeFrameGPU->downloadTrackITSExtDevice();

    auto& tracks = mTimeFrameGPU->getTrackITSExt();

    for (auto& track : tracks) {
      if (!track.getChi2()) {
        continue; // this is to skip the unset tracks that are put at the beginning of the vector by the sorting. To see if this can be optimised.
      }
      int nShared = 0;
      bool isFirstShared{false};
      for (int iLayer{0}; iLayer < this->mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
          continue;
        }
        nShared += int(mTimeFrameGPU->isClusterUsed(iLayer, track.getClusterIndex(iLayer)));
        isFirstShared |= !iLayer && mTimeFrameGPU->isClusterUsed(iLayer, track.getClusterIndex(iLayer));
      }

      if (nShared > this->mTrkParams[0].ClusterSharing) {
        continue;
      }

      bool firstCls{true};
      TimeEstBC ts;
      for (int iLayer{0}; iLayer < this->mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
          continue;
        }
        mTimeFrameGPU->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        int currentROF = mTimeFrameGPU->getClusterROF(iLayer, track.getClusterIndex(iLayer));
        auto rofTS = mTimeFrameGPU->getROFOverlapTableView().getLayer(iLayer).getROFTimeBounds(currentROF, true);
        if (firstCls) {
          ts = rofTS;
        } else {
          if (!ts.isCompatible(rofTS)) {
            LOGP(fatal, "TS {}+/-{} are incompatible with {}+/-{}, this should not happen!", rofTS.getTimeStamp(), rofTS.getTimeStampError(), ts.getTimeStamp(), ts.getTimeStampError());
          }
          ts += rofTS;
        }
      }
      track.getTimeStamp() = ts.makeSymmetrical();
      track.setUserField(0);
      track.getParamOut().setUserField(0);
      mTimeFrameGPU->getTracks().emplace_back(track);
    }
    mTimeFrameGPU->loadUsedClustersDevice();
  }
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
} // namespace o2::its
