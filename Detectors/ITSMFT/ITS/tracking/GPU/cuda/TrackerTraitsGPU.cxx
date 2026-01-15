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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::initialiseTimeFrame(const int iteration)
{
  mTimeFrameGPU->initialise(iteration, this->mTrkParams[iteration], nLayers);
  // on default stream
  mTimeFrameGPU->loadVertices(iteration);
  mTimeFrameGPU->loadIndexTableUtils(iteration);
  mTimeFrameGPU->loadMultiplicityCutMask(iteration);
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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::adoptTimeFrame(TimeFrame<nLayers>* tf)
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<nLayers>*>(tf);
  this->mTimeFrame = static_cast<TimeFrame<nLayers>*>(tf);
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerTracklets(const int iteration, int iROFslice, int iVertex)
{
  const auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  int startROF{0};
  int endROF{mTimeFrameGPU->getNrof()};

  // start by queuing loading needed of two last layers
  for (int iLayer{nLayers}; iLayer-- > nLayers - 2;) {
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
    countTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceMultCutMask(),
                                         iLayer,
                                         startROF,
                                         endROF,
                                         mTimeFrameGPU->getNrof(),
                                         this->mTrkParams[iteration].DeltaROF,
                                         iVertex,
                                         mTimeFrameGPU->getDeviceVertices(),
                                         mTimeFrameGPU->getDeviceROFramesPV(),
                                         mTimeFrameGPU->getPrimaryVerticesNum(),
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
    computeTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                           mTimeFrameGPU->getDeviceMultCutMask(),
                                           iLayer,
                                           startROF,
                                           endROF,
                                           mTimeFrameGPU->getNrof(),
                                           this->mTrkParams[iteration].DeltaROF,
                                           iVertex,
                                           mTimeFrameGPU->getDeviceVertices(),
                                           mTimeFrameGPU->getDeviceROFramesPV(),
                                           mTimeFrameGPU->getPrimaryVerticesNum(),
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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerCells(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  // start by queuing loading needed of three last layers
  for (int iLayer{nLayers}; iLayer-- > nLayers - 3;) {
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
    countCellsHandler<nLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                               mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                               mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                               mTimeFrameGPU->getDeviceArrayTracklets(),
                               mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                               currentLayerTrackletsNum,
                               iLayer,
                               nullptr,
                               mTimeFrameGPU->getDeviceArrayCellsLUT(),
                               mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                               this->mTrkParams[iteration].DeltaROF,
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
    computeCellsHandler<nLayers>(mTimeFrameGPU->getDeviceArrayClusters(),
                                 mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                                 mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                 mTimeFrameGPU->getDeviceArrayTracklets(),
                                 mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                 currentLayerTrackletsNum,
                                 iLayer,
                                 mTimeFrameGPU->getDeviceCells()[iLayer],
                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                 mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                                 this->mTrkParams[iteration].DeltaROF,
                                 this->mBz,
                                 this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                                 this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                                 this->mTrkParams[iteration].NSigmaCut,
                                 conf.nBlocksLayerCells[iteration],
                                 conf.nThreadsLayerCells[iteration],
                                 mTimeFrameGPU->getStreams());
  }
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration)
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
    countCellNeighboursHandler<nLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                        mTimeFrameGPU->getDeviceNeighboursLUT(iLayer), // LUT is initialised here.
                                        mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                        mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                        mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                        (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                        this->mTrkParams[0].DeltaROF,
                                        this->mTrkParams[0].MaxChi2ClusterAttachment,
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
    computeCellNeighboursHandler<nLayers>(mTimeFrameGPU->getDeviceArrayCells(),
                                          mTimeFrameGPU->getDeviceNeighboursLUT(iLayer),
                                          mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                          mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                          mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                          (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                          this->mTrkParams[0].DeltaROF,
                                          this->mTrkParams[0].MaxChi2ClusterAttachment,
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

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    bounded_vector<CellSeed<nLayers>> trackSeeds(this->getMemoryPool().get());
    for (int startLayer{this->mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      if ((this->mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }
      processNeighboursHandler<nLayers>(startLayer,
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
    mTimeFrameGPU->createTrackITSExtDevice(trackSeeds);
    mTimeFrameGPU->loadTrackSeedsDevice(trackSeeds);

    trackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),             // CellSeed*
                     mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo**
                     mTimeFrameGPU->getDeviceArrayUnsortedClusters(),  // Cluster**
                     mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt*
                     this->mTrkParams[iteration].LayerRadii,           // const std::vector<float>&
                     this->mTrkParams[iteration].MinPt,                // const std::vector<float>&
                     trackSeeds.size(),                                // const size_t nSeeds
                     this->mBz,                                        // const float Bz
                     startLevel,                                       // const int startLevel,
                     this->mTrkParams[0].MaxChi2ClusterAttachment,     // float maxChi2ClusterAttachment
                     this->mTrkParams[0].MaxChi2NDF,                   // float maxChi2NDF
                     this->mTrkParams[0].RepeatRefitOut,
                     this->mTrkParams[0].ReseedIfShorter,
                     this->mTrkParams[0].ShiftRefToCluster,
                     mTimeFrameGPU->getDevicePropagator(), // const o2::base::Propagator* propagator
                     this->mTrkParams[0].CorrType,         // o2::base::PropagatorImpl<float>::MatCorrType
                     conf.nBlocksTracksSeeds[iteration],
                     conf.nThreadsTracksSeeds[iteration]);

    mTimeFrameGPU->downloadTrackITSExtDevice(trackSeeds);

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

      std::array<int, 3> rofs{INT_MAX, INT_MAX, INT_MAX};
      for (int iLayer{0}; iLayer < this->mTrkParams[0].NLayers; ++iLayer) {
        if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
          continue;
        }
        mTimeFrameGPU->markUsedCluster(iLayer, track.getClusterIndex(iLayer));
        int currentROF = mTimeFrameGPU->getClusterROF(iLayer, track.getClusterIndex(iLayer));
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
      mTimeFrameGPU->getTracks(std::min(rofs[0], rofs[1])).emplace_back(track);
    }
    mTimeFrameGPU->loadUsedClustersDevice();
  }
  // wipe the artefact memory
  mTimeFrameGPU->popMemoryStack(iteration);
};

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfClusters() const
{
  return mTimeFrameGPU->getNumberOfClusters();
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfTracklets() const
{
  return std::accumulate(mTimeFrameGPU->getNTracklets().begin(), mTimeFrameGPU->getNTracklets().end(), 0);
}

template <int nLayers>
int TrackerTraitsGPU<nLayers>::getTFNumberOfCells() const
{
  return mTimeFrameGPU->getNumberOfCells();
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::setBz(float bz)
{
  this->mBz = bz;
  mTimeFrameGPU->setBz(bz);
}

template class TrackerTraitsGPU<7>;
} // namespace o2::its
