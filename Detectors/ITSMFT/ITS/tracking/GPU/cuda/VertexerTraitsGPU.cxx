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

#include <gsl/span>

#include "ITStracking/TrackingConfigParam.h"
#include "ITStrackingGPU/VertexingKernels.h"
#include "ITStrackingGPU/VertexerTraitsGPU.h"

namespace o2::its
{

template <int nLayers>
void VertexerTraitsGPU<nLayers>::initialise(const TrackingParameters& trackingParams, const int iteration)
{
  // FIXME
  // Two things to fix here:
  // This loads all necessary data for this step at once, can be overlayed with computation
  // Also if running with the tracker some data is loaded twice!
  mTimeFrameGPU->initialise(0, trackingParams, 3, &this->mIndexTableUtils, &mTfGPUParams);

  // FIXME some of these only need to be created once!
  mTimeFrameGPU->loadIndexTableUtils(iteration);
  mTimeFrameGPU->createUsedClustersDeviceArray(iteration, 3);
  mTimeFrameGPU->createClustersDeviceArray(iteration, 3);
  mTimeFrameGPU->createUnsortedClustersDeviceArray(iteration, 3);
  mTimeFrameGPU->createClustersIndexTablesArray(iteration);
  mTimeFrameGPU->createROFrameClustersDeviceArray(iteration);
  for (int iLayer{0}; iLayer < 3; ++iLayer) {
    mTimeFrameGPU->loadClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadUnsortedClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadClustersIndexTables(iteration, iLayer);
    mTimeFrameGPU->createUsedClustersDevice(iteration, iLayer);
    mTimeFrameGPU->loadROFrameClustersDevice(iteration, iLayer);
  }
}

template <int nLayers>
void VertexerTraitsGPU<nLayers>::adoptTimeFrame(TimeFrame<nLayers>* tf) noexcept
{
  mTimeFrameGPU = static_cast<gpu::TimeFrameGPU<nLayers>*>(tf);
  this->mTimeFrame = static_cast<TimeFrame<nLayers>*>(tf);
}

template <int nLayers>
void VertexerTraitsGPU<nLayers>::updateVertexingParameters(const std::vector<VertexingParameters>& vrtPar, const TimeFrameGPUParameters& tfPar)
{
  this->mVrtParams = vrtPar;
  mTfGPUParams = tfPar;
  this->mIndexTableUtils.setTrackingParameters(vrtPar[0]);
  for (auto& par : this->mVrtParams) {
    par.phiSpan = static_cast<int>(std::ceil(this->mIndexTableUtils.getNphiBins() * par.phiCut / o2::constants::math::TwoPI));
    par.zSpan = static_cast<int>(std::ceil(par.zCut * this->mIndexTableUtils.getInverseZCoordinate(0)));
  }
}

template <int nLayers>
void VertexerTraitsGPU<nLayers>::computeTracklets(const int iteration)
{
  if (mTimeFrameGPU->getClusters().empty()) {
    return;
  }
  const auto& conf = ITSGpuTrackingParamConfig::Instance();

  mTimeFrameGPU->createVtxTrackletsLUTDevice(iteration);
  countTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                       mTimeFrameGPU->getDeviceMultCutMask(),
                                       mTimeFrameGPU->getNrof(),
                                       this->mVrtParams[iteration].deltaRof,
                                       mTimeFrameGPU->getDeviceROFramesPV(),
                                       this->mVrtParams[iteration].vertPerRofThreshold,
                                       mTimeFrameGPU->getDeviceArrayClusters(),
                                       mTimeFrameGPU->getClusterSizes()[1],
                                       mTimeFrameGPU->getDeviceROFrameClusters(),
                                       (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                       mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                       mTimeFrameGPU->getDeviceArrayNTrackletsPerCluster(),
                                       mTimeFrameGPU->getDeviceArrayNTrackletsPerClusterSum(),
                                       mTimeFrameGPU->getDeviceArrayNTrackletsPerROF(),
                                       mTimeFrameGPU->getDeviceNTrackletsPerCluster(),
                                       mTimeFrameGPU->getDeviceNTrackletsPerClusterSum(),
                                       iteration,
                                       this->mVrtParams[iteration].phiCut,
                                       this->mVrtParams[iteration].maxTrackletsPerCluster,
                                       conf.nBlocksVtxComputeTracklets[iteration],
                                       conf.nThreadsVtxComputeTracklets[iteration],
                                       mTimeFrameGPU->getStreams());
  mTimeFrameGPU->createVtxTrackletsBuffers(iteration);
  computeTrackletsInROFsHandler<nLayers>(mTimeFrameGPU->getDeviceIndexTableUtils(),
                                         mTimeFrameGPU->getDeviceMultCutMask(),
                                         mTimeFrameGPU->getNrof(),
                                         this->mVrtParams[iteration].deltaRof,
                                         mTimeFrameGPU->getDeviceROFramesPV(),
                                         this->mVrtParams[iteration].vertPerRofThreshold,
                                         mTimeFrameGPU->getDeviceArrayClusters(),
                                         mTimeFrameGPU->getClusterSizes()[1],
                                         mTimeFrameGPU->getDeviceROFrameClusters(),
                                         (const uint8_t**)mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                         mTimeFrameGPU->getDeviceArrayClustersIndexTables(),
                                         mTimeFrameGPU->getDeviceArrayTracklets(),
                                         (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerCluster(),
                                         (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerClusterSum(),
                                         (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerROF(),
                                         iteration,
                                         this->mVrtParams[iteration].phiCut,
                                         this->mVrtParams[iteration].maxTrackletsPerCluster,
                                         conf.nBlocksVtxComputeTracklets[iteration],
                                         conf.nThreadsVtxComputeTracklets[iteration],
                                         mTimeFrameGPU->getStreams());
}

template <int nLayers>
void VertexerTraitsGPU<nLayers>::computeTrackletMatching(const int iteration)
{
  if (!mTimeFrameGPU->getTotalTrackletsTF(0) || !mTimeFrameGPU->getTotalTrackletsTF(1)) {
    return;
  }

  const auto& conf = ITSGpuTrackingParamConfig::Instance();
  mTimeFrameGPU->createVtxLinesLUTDevice(iteration);
  countTrackletsMatchingInROFsHandler(mTimeFrameGPU->getNrof(),
                                      this->mVrtParams[iteration].deltaRof,
                                      mTimeFrameGPU->getClusterSizes()[1],
                                      mTimeFrameGPU->getDeviceROFrameClusters(),
                                      mTimeFrameGPU->getDeviceArrayClusters(),
                                      mTimeFrameGPU->getDeviceArrayUsedClusters(),
                                      (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                      mTimeFrameGPU->getDeviceUsedTracklets(),
                                      (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerCluster(),
                                      (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerClusterSum(),
                                      mTimeFrameGPU->getDeviceNLinesPerCluster(),
                                      mTimeFrameGPU->getDeviceNLinesPerClusterSum(),
                                      iteration,
                                      this->mVrtParams[iteration].phiCut,
                                      this->mVrtParams[iteration].tanLambdaCut,
                                      conf.nBlocksVtxComputeMatching[iteration],
                                      conf.nThreadsVtxComputeMatching[iteration],
                                      mTimeFrameGPU->getStreams());
  mTimeFrameGPU->createVtxLinesBuffer(iteration);
  computeTrackletsMatchingInROFsHandler(mTimeFrameGPU->getNrof(),
                                        this->mVrtParams[iteration].deltaRof,
                                        mTimeFrameGPU->getClusterSizes()[1],
                                        mTimeFrameGPU->getDeviceROFrameClusters(),
                                        mTimeFrameGPU->getDeviceArrayClusters(),
                                        nullptr,
                                        (const Tracklet**)mTimeFrameGPU->getDeviceArrayTracklets(),
                                        mTimeFrameGPU->getDeviceUsedTracklets(),
                                        (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerCluster(),
                                        (const int32_t**)mTimeFrameGPU->getDeviceArrayNTrackletsPerClusterSum(),
                                        (const int32_t*)mTimeFrameGPU->getDeviceNLinesPerClusterSum(),
                                        mTimeFrameGPU->getDeviceLines(),
                                        iteration,
                                        this->mVrtParams[iteration].phiCut,
                                        this->mVrtParams[iteration].tanLambdaCut,
                                        conf.nBlocksVtxComputeMatching[iteration],
                                        conf.nThreadsVtxComputeMatching[iteration],
                                        mTimeFrameGPU->getStreams());
}

template <int nLayers>
void VertexerTraitsGPU<nLayers>::computeVertices(const int iteration)
{
  LOGP(fatal, "This step is not implemented yet!");
  mTimeFrameGPU->loadUsedClustersDevice();
}

template class VertexerTraitsGPU<7>;

} // namespace o2::its
