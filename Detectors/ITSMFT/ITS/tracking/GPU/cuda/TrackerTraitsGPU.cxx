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
  mTimeFrameGPU->loadClustersDevice(iteration);
  mTimeFrameGPU->loadUnsortedClustersDevice(iteration);
  mTimeFrameGPU->loadClustersIndexTables(iteration);
  mTimeFrameGPU->loadTrackingFrameInfoDevice(iteration);
  mTimeFrameGPU->loadMultiplicityCutMask(iteration);
  mTimeFrameGPU->loadVertices(iteration);
  mTimeFrameGPU->loadROframeClustersDevice(iteration);
  mTimeFrameGPU->createUsedClustersDevice(iteration);
  mTimeFrameGPU->loadIndexTableUtils(iteration);
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

  int startROF{this->mTrkParams[iteration].nROFsPerIterations > 0 ? iROFslice * this->mTrkParams[iteration].nROFsPerIterations : 0};
  int endROF{o2::gpu::CAMath::Min(this->mTrkParams[iteration].nROFsPerIterations > 0 ? (iROFslice + 1) * this->mTrkParams[iteration].nROFsPerIterations + this->mTrkParams[iteration].DeltaROF : mTimeFrameGPU->getNrof(), mTimeFrameGPU->getNrof())};

  mTimeFrameGPU->createTrackletsLUTDevice(iteration);
  countTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                       mTimeFrameGPU->getDeviceMultCutMask(),
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
                                       mTimeFrameGPU->getDeviceROframeClusters(),
                                       mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                       mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                       mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                                       mTimeFrameGPU->getDeviceTrackletsLUTs(), // Required for the exclusive sums
                                       iteration,
                                       this->mTrkParams[iteration].NSigmaCut,
                                       mTimeFrameGPU->getPhiCuts(),
                                       this->mTrkParams[iteration].PVres,
                                       mTimeFrameGPU->getMinRs(),
                                       mTimeFrameGPU->getMaxRs(),
                                       mTimeFrameGPU->getPositionResolutions(),
                                       this->mTrkParams[iteration].LayerRadii,
                                       mTimeFrameGPU->getMSangles(),
                                       conf.nBlocksLayerTracklets[iteration],
                                       conf.nThreadsLayerTracklets[iteration],
                                       mTimeFrameGPU->getStreams());
  mTimeFrameGPU->createTrackletsBuffers();
  computeTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceMultCutMask(),
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
                                         mTimeFrameGPU->getDeviceROframeClusters(),
                                         mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                         mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                         mTimeFrameGPU->getDeviceArrayTracklets(),
                                         mTimeFrameGPU->getDeviceTracklet(),
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
                                         conf.nBlocksLayerTracklets[iteration],
                                         conf.nThreadsLayerTracklets[iteration],
                                         mTimeFrameGPU->getStreams());
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::computeLayerCells(const int iteration)
{
  mTimeFrameGPU->createCellsLUTDevice();
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();

  for (int iLayer = 0; iLayer < this->mTrkParams[iteration].CellsPerRoad(); ++iLayer) {
    if (!mTimeFrameGPU->getNTracklets()[iLayer + 1] || !mTimeFrameGPU->getNTracklets()[iLayer]) {
      mTimeFrameGPU->getNCells()[iLayer] = 0;
      continue;
    }
    const int currentLayerTrackletsNum{static_cast<int>(mTimeFrameGPU->getNTracklets()[iLayer])};
    countCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                      mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                      mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                      mTimeFrameGPU->getDeviceArrayTracklets(),
                      mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                      mTimeFrameGPU->getNTracklets()[iLayer],
                      iLayer,
                      nullptr,
                      mTimeFrameGPU->getDeviceArrayCellsLUT(),
                      mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                      this->mBz,
                      this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                      this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                      this->mTrkParams[iteration].NSigmaCut,
                      conf.nBlocksLayerCells[iteration],
                      conf.nThreadsLayerCells[iteration]);
    mTimeFrameGPU->createCellsBuffers(iLayer);
    computeCellsHandler(mTimeFrameGPU->getDeviceArrayClusters(),
                        mTimeFrameGPU->getDeviceArrayUnsortedClusters(),
                        mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                        mTimeFrameGPU->getDeviceArrayTracklets(),
                        mTimeFrameGPU->getDeviceArrayTrackletsLUT(),
                        mTimeFrameGPU->getNTracklets()[iLayer],
                        iLayer,
                        mTimeFrameGPU->getDeviceCells()[iLayer],
                        mTimeFrameGPU->getDeviceArrayCellsLUT(),
                        mTimeFrameGPU->getDeviceCellLUTs()[iLayer],
                        this->mBz,
                        this->mTrkParams[iteration].MaxChi2ClusterAttachment,
                        this->mTrkParams[iteration].CellDeltaTanLambdaSigma,
                        this->mTrkParams[iteration].NSigmaCut,
                        conf.nBlocksLayerCells[iteration],
                        conf.nThreadsLayerCells[iteration]);
  }
}

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findCellsNeighbours(const int iteration)
{
  mTimeFrameGPU->createNeighboursIndexTablesDevice();
  const auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int iLayer{0}; iLayer < this->mTrkParams[iteration].CellsPerRoad() - 1; ++iLayer) {
    const int currentLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer])};
    const int nextLayerCellsNum{static_cast<int>(mTimeFrameGPU->getNCells()[iLayer + 1])};
    if (!nextLayerCellsNum || !currentLayerCellsNum) {
      mTimeFrameGPU->getNNeighbours()[iLayer] = 0;
      continue;
    }

    mTimeFrameGPU->createNeighboursLUTDevice(iLayer, nextLayerCellsNum);
    unsigned int nNeigh = countCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                                     mTimeFrameGPU->getDeviceNeighboursLUT(iLayer), // LUT is initialised here.
                                                     mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                                     mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                                     mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                                     this->mTrkParams[0].MaxChi2ClusterAttachment,
                                                     this->mBz,
                                                     iLayer,
                                                     currentLayerCellsNum,
                                                     nextLayerCellsNum,
                                                     1e2,
                                                     conf.nBlocksFindNeighbours[iteration],
                                                     conf.nThreadsFindNeighbours[iteration]);

    mTimeFrameGPU->createNeighboursDevice(iLayer, nNeigh);

    computeCellNeighboursHandler(mTimeFrameGPU->getDeviceArrayCells(),
                                 mTimeFrameGPU->getDeviceNeighboursLUT(iLayer),
                                 mTimeFrameGPU->getDeviceArrayCellsLUT(),
                                 mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                 mTimeFrameGPU->getDeviceNeighboursIndexTables(iLayer),
                                 this->mTrkParams[0].MaxChi2ClusterAttachment,
                                 this->mBz,
                                 iLayer,
                                 currentLayerCellsNum,
                                 nextLayerCellsNum,
                                 1e2,
                                 conf.nBlocksFindNeighbours[iteration],
                                 conf.nThreadsFindNeighbours[iteration]);

    nNeigh = filterCellNeighboursHandler(mTimeFrameGPU->getDeviceNeighbourPairs(iLayer),
                                         mTimeFrameGPU->getDeviceNeighbours(iLayer),
                                         nNeigh,
                                         mTimeFrameGPU->getExternalAllocator());
    mTimeFrameGPU->getArrayNNeighbours()[iLayer] = nNeigh;
  }
  mTimeFrameGPU->createNeighboursDeviceArray();
  mTimeFrameGPU->unregisterRest();
};

template <int nLayers>
void TrackerTraitsGPU<nLayers>::findRoads(const int iteration)
{
  auto& conf = o2::its::ITSGpuTrackingParamConfig::Instance();
  for (int startLevel{this->mTrkParams[iteration].CellsPerRoad()}; startLevel >= this->mTrkParams[iteration].CellMinimumLevel(); --startLevel) {
    const int minimumLayer{startLevel - 1};
    bounded_vector<CellSeed> trackSeeds(this->getMemoryPool().get());
    for (int startLayer{this->mTrkParams[iteration].CellsPerRoad() - 1}; startLayer >= minimumLayer; --startLayer) {
      if ((this->mTrkParams[iteration].StartLayerMask & (1 << (startLayer + 2))) == 0) {
        continue;
      }
      processNeighboursHandler<nLayers>(startLayer,
                                        startLevel,
                                        mTimeFrameGPU->getDeviceArrayCells(),
                                        mTimeFrameGPU->getDeviceCells()[startLayer],
                                        mTimeFrameGPU->getArrayNCells(),
                                        mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                        mTimeFrameGPU->getDeviceNeighboursAll(),
                                        mTimeFrameGPU->getDeviceNeighboursLUTs(),
                                        mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(),
                                        trackSeeds,
                                        mTimeFrameGPU->getExternalAllocator(),
                                        this->mBz,
                                        this->mTrkParams[0].MaxChi2ClusterAttachment,
                                        this->mTrkParams[0].MaxChi2NDF,
                                        mTimeFrameGPU->getDevicePropagator(),
                                        this->mTrkParams[0].CorrType,
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

    trackSeedHandler(mTimeFrameGPU->getDeviceTrackSeeds(),             // CellSeed* trackSeeds
                     mTimeFrameGPU->getDeviceArrayTrackingFrameInfo(), // TrackingFrameInfo** foundTrackingFrameInfo
                     mTimeFrameGPU->getDeviceTrackITSExt(),            // o2::its::TrackITSExt* tracks
                     this->mTrkParams[iteration].MinPt,                // std::vector<float>& minPtsHost,
                     trackSeeds.size(),                                // const size_t nSeeds
                     this->mBz,                                        // const float Bz
                     startLevel,                                       // const int startLevel,
                     this->mTrkParams[0].MaxChi2ClusterAttachment,     // float maxChi2ClusterAttachment
                     this->mTrkParams[0].MaxChi2NDF,                   // float maxChi2NDF
                     mTimeFrameGPU->getDevicePropagator(),             // const o2::base::Propagator* propagator
                     this->mTrkParams[0].CorrType,                     // o2::base::PropagatorImpl<float>::MatCorrType
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
