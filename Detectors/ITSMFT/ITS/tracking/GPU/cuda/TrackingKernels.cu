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

#include <cuda_runtime.h>
#include <array>
#include <unistd.h>

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/remove.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/IndexTableUtils.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/ExternalAllocator.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"
#include "ITStracking/Cell.h"
#include "DataFormatsITS/TrackITS.h"
#include "ITStrackingGPU/TrackingKernels.h"
#include "ITStrackingGPU/Utils.h"
#include "utils/strtag.h"

// O2 track model
#include "ReconstructionDataFormats/Track.h"
#include "DetectorsBase/Propagator.h"
using namespace o2::track;

namespace o2::its
{
namespace gpu
{

GPUdii() bool fitTrack(TrackITSExt& track,
                       int start,
                       int end,
                       int step,
                       float chi2clcut,
                       float chi2ndfcut,
                       float maxQoverPt,
                       int nCl,
                       float bz,
                       const TrackingFrameInfo** tfInfos,
                       const o2::base::Propagator* prop,
                       o2::base::PropagatorF::MatCorrType matCorrType,
                       o2::track::TrackPar* linRef,
                       const bool shiftRefToCluster)
{
  for (int iLayer{start}; iLayer != end; iLayer += step) {
    if (track.getClusterIndex(iLayer) == constants::UnusedIndex) {
      continue;
    }
    const TrackingFrameInfo& trackingHit = tfInfos[iLayer][track.getClusterIndex(iLayer)];
    if (linRef) {
      if (!track.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame, *linRef, bz)) {
        return false;
      }
      if (!prop->propagateToX(track,
                              *linRef,
                              trackingHit.xTrackingFrame,
                              bz,
                              o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                              o2::base::PropagatorImpl<float>::MAX_STEP,
                              matCorrType)) {

        return false;
      }
      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        const float xx0 = (iLayer > 2) ? 1.e-2f : 5.e-3f; // Rough layer thickness
        if (!track.correctForMaterial(*linRef, xx0, xx0 * constants::Radl * constants::Rho, true)) {
          return false;
        }
      }
    } else {
      if (!track.o2::track::TrackParCovF::rotate(trackingHit.alphaTrackingFrame)) {
        return false;
      }
      if (!prop->propagateToX(track,
                              trackingHit.xTrackingFrame,
                              bz,
                              o2::base::PropagatorImpl<float>::MAX_SIN_PHI,
                              o2::base::PropagatorImpl<float>::MAX_STEP,
                              matCorrType)) {
        return false;
      }
      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        const float xx0 = (iLayer > 2) ? 1.e-2f : 5.e-3f; // Rough layer thickness
        if (!track.correctForMaterial(xx0, xx0 * constants::Radl * constants::Rho, true)) {
          return false;
        }
      }
    }

    auto predChi2{track.getPredictedChi2(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
    if ((nCl >= 3 && predChi2 > chi2clcut) || predChi2 < 0.f) {
      return false;
    }
    track.setChi2(track.getChi2() + predChi2);
    if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
      return false;
    }
    if (linRef && shiftRefToCluster) { // displace the reference to the last updated cluster
      linRef->setY(trackingHit.positionTrackingFrame[0]);
      linRef->setZ(trackingHit.positionTrackingFrame[1]);
    }
    nCl++;
  }
  return o2::gpu::CAMath::Abs(track.getQ2Pt()) < maxQoverPt && track.getChi2() < chi2ndfcut * (nCl * 2 - 5);
}

GPUdii() o2::track::TrackParCov buildTrackSeed(const Cluster& cluster1,
                                               const Cluster& cluster2,
                                               const TrackingFrameInfo& tf3,
                                               const float bz,
                                               const bool reverse = false)
{
  const float sign = reverse ? -1.f : 1.f;

  float ca, sa;
  o2::gpu::CAMath::SinCos(tf3.alphaTrackingFrame, sa, ca);

  const float x1 = cluster1.xCoordinate * ca + cluster1.yCoordinate * sa;
  const float y1 = -cluster1.xCoordinate * sa + cluster1.yCoordinate * ca;
  const float x2 = cluster2.xCoordinate * ca + cluster2.yCoordinate * sa;
  const float y2 = -cluster2.xCoordinate * sa + cluster2.yCoordinate * ca;
  const float x3 = tf3.xTrackingFrame;
  const float y3 = tf3.positionTrackingFrame[0];

  float snp, q2pt, q2pt2;
  if (o2::gpu::CAMath::Abs(bz) < 0.01f) {
    const float dx = x3 - x1;
    const float dy = y3 - y1;
    snp = sign * dy / o2::gpu::CAMath::Hypot(dx, dy);
    q2pt = 1.f / track::kMostProbablePt;
    q2pt2 = 1.f;
  } else {
    const float crv = math_utils::computeCurvature(x3, y3, x2, y2, x1, y1);
    snp = sign * crv * (x3 - math_utils::computeCurvatureCentreX(x3, y3, x2, y2, x1, y1));
    q2pt = sign * crv / (bz * o2::constants::math::B2C);
    q2pt2 = crv * crv;
  }

  const float tgl = 0.5f * (math_utils::computeTanDipAngle(x1, y1, x2, y2, cluster1.zCoordinate, cluster2.zCoordinate) +
                            math_utils::computeTanDipAngle(x2, y2, x3, y3, cluster2.zCoordinate, tf3.positionTrackingFrame[1]));
  const float sg2q2pt = track::kC1Pt2max * (q2pt2 > 0.0005f ? (q2pt2 < 1.f ? q2pt2 : 1.f) : 0.0005f);

  return {x3, tf3.alphaTrackingFrame, {y3, tf3.positionTrackingFrame[1], snp, tgl, q2pt}, {tf3.covarianceTrackingFrame[0], tf3.covarianceTrackingFrame[1], tf3.covarianceTrackingFrame[2], 0.f, 0.f, track::kCSnp2max, 0.f, 0.f, 0.f, track::kCTgl2max, 0.f, 0.f, 0.f, 0.f, sg2q2pt}};
}

template <int NLayers>
GPUdii() TrackITSExt seedTrackForRefit(const CellSeed<NLayers>& seed,
                                       const TrackingFrameInfo** foundTrackingFrameInfo,
                                       const Cluster** unsortedClusters,
                                       const float* layerRadii,
                                       const float bz,
                                       const int reseedIfShorter)
{
  TrackITSExt temporaryTrack(seed);
  int lrMin = NLayers, lrMax = 0, lrMid = 0;
  for (int iL{0}; iL < NLayers; ++iL) {
    const int idx = seed.getCluster(iL);
    temporaryTrack.setExternalClusterIndex(iL, idx, idx != constants::UnusedIndex);
    if (idx != constants::UnusedIndex) {
      // TODO only works if does not have holes
      lrMin = o2::gpu::CAMath::Min(lrMin, iL);
      lrMax = o2::gpu::CAMath::Max(lrMax, iL);
    }
  }
  const int ncl = temporaryTrack.getNClusters();
  if (ncl < reseedIfShorter && ncl > 0) { // need to check if there are any clusters since we keep invalidate seeeds around
    if (ncl == NLayers) {
      lrMin = 0;
      lrMax = NLayers - 1;
      lrMid = (lrMin + lrMax) / 2;
    } else {
      lrMid = lrMin + 1;
      float midR = 0.5f * (layerRadii[lrMax] + layerRadii[lrMin]), dstMidR = o2::gpu::CAMath::Abs(midR - layerRadii[lrMid]);
      for (int iL = lrMid + 1; iL < lrMax; ++iL) { // find the midpoint as closest to the midR
        auto dst = o2::gpu::GPUCommonMath::Abs(midR - layerRadii[iL]);
        if (dst < dstMidR) {
          lrMid = iL;
          dstMidR = dst;
        }
      }
    }
    const auto& cluster0_tf = foundTrackingFrameInfo[lrMin][seed.getCluster(lrMin)];
    const auto& cluster1_gl = unsortedClusters[lrMid][seed.getCluster(lrMid)];
    const auto& cluster2_gl = unsortedClusters[lrMax][seed.getCluster(lrMax)];
    temporaryTrack.getParamIn() = buildTrackSeed(cluster2_gl, cluster1_gl, cluster0_tf, bz, true);
  }
  temporaryTrack.resetCovariance();
  temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
  return temporaryTrack;
}

struct sort_tracklets {
  GPUhd() bool operator()(const Tracklet& a, const Tracklet& b)
  {
    if (a.firstClusterIndex != b.firstClusterIndex) {
      return a.firstClusterIndex < b.firstClusterIndex;
    }
    return a.secondClusterIndex < b.secondClusterIndex;
  }
};

struct equal_tracklets {
  GPUhd() bool operator()(const Tracklet& a, const Tracklet& b) { return a.firstClusterIndex == b.firstClusterIndex && a.secondClusterIndex == b.secondClusterIndex; }
};

template <typename T1, typename T2>
struct sort_by_second {
  GPUhd() bool operator()(const gpuPair<T1, T2>& a, const gpuPair<T1, T2>& b) const { return a.second < b.second; }
};

template <typename T1, typename T2>
struct pair_to_first {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.first;
  }
};

template <typename T1, typename T2>
struct pair_to_second {
  GPUhd() int operator()(const gpuPair<T1, T2>& a) const
  {
    return a.second;
  }
};

template <typename T1, typename T2>
struct is_invalid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return p.first == -1 && p.second == -1;
  }
};

template <typename T1, typename T2>
struct is_valid_pair {
  GPUhd() bool operator()(const gpuPair<T1, T2>& p) const
  {
    return !(p.first == -1 && p.second == -1);
  }
};

template <int NLayers>
struct seed_selector {
  float maxQ2Pt;
  float maxChi2;

  GPUhd() seed_selector(float maxQ2Pt, float maxChi2) : maxQ2Pt(maxQ2Pt), maxChi2(maxChi2) {}
  GPUhd() bool operator()(const CellSeed<NLayers>& seed) const
  {
    return !(seed.getQ2Pt() > maxQ2Pt || seed.getChi2() > maxChi2);
  }
};

struct compare_track_chi2 {
  GPUhd() bool operator()(const TrackITSExt& a, const TrackITSExt& b) const
  {
    return a.getChi2() < b.getChi2();
  }
};

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(256, 1) fitTrackSeedsKernel(
  CellSeed<NLayers>* trackSeeds,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const Cluster** unsortedClusters,
  o2::its::TrackITSExt* tracks,
  maybe_const<!initRun, int>* seedLUT,
  const float* layerRadii,
  const float* minPts,
  const unsigned int nSeeds,
  const float bz,
  const int startLevel,
  const float maxChi2ClusterAttachment,
  const float maxChi2NDF,
  const int reseedIfShorter,
  const bool repeatRefitOut,
  const bool shifRefToCluster,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  for (int iCurrentTrackSeedIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackSeedIndex < nSeeds; iCurrentTrackSeedIndex += blockDim.x * gridDim.x) {

    if constexpr (!initRun) {
      if (seedLUT[iCurrentTrackSeedIndex] == seedLUT[iCurrentTrackSeedIndex + 1]) {
        continue;
      }
    }

    TrackITSExt temporaryTrack = seedTrackForRefit<NLayers>(trackSeeds[iCurrentTrackSeedIndex], foundTrackingFrameInfo, unsortedClusters, layerRadii, bz, reseedIfShorter);
    o2::track::TrackPar linRef{temporaryTrack};
    bool fitSuccess = fitTrack(temporaryTrack,               // TrackITSExt& track,
                               0,                            // int lastLayer,
                               NLayers,                      // int firstLayer,
                               1,                            // int firstCluster,
                               maxChi2ClusterAttachment,     // float maxChi2ClusterAttachment,
                               maxChi2NDF,                   // float maxChi2NDF,
                               o2::constants::math::VeryBig, // float maxQoverPt,
                               0,                            // nCl,
                               bz,                           // float bz,
                               foundTrackingFrameInfo,       // TrackingFrameInfo** trackingFrameInfo,
                               propagator,                   // const o2::base::Propagator* propagator,
                               matCorrType,                  // o2::base::PropagatorF::MatCorrType matCorrType
                               &linRef,
                               shifRefToCluster);
    if (!fitSuccess) {
      continue;
    }
    temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
    linRef = temporaryTrack.getParamOut(); // use refitted track as lin.reference
    temporaryTrack.resetCovariance();
    temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
    temporaryTrack.setChi2(0);
    fitSuccess = fitTrack(temporaryTrack,           // TrackITSExt& track,
                          NLayers - 1,              // int lastLayer,
                          -1,                       // int firstLayer,
                          -1,                       // int firstCluster,
                          maxChi2ClusterAttachment, // float maxChi2ClusterAttachment,
                          maxChi2NDF,               // float maxChi2NDF,
                          50.f,                     // float maxQoverPt,
                          0,                        // nCl,
                          bz,                       // float bz,
                          foundTrackingFrameInfo,   // TrackingFrameInfo** trackingFrameInfo,
                          propagator,               // const o2::base::Propagator* propagator,
                          matCorrType,              // o2::base::PropagatorF::MatCorrType matCorrType
                          &linRef,
                          shifRefToCluster);
    if (!fitSuccess || temporaryTrack.getPt() < minPts[NLayers - temporaryTrack.getNClusters()]) {
      continue;
    }
    if (repeatRefitOut) { // repeat outward refit seeding and linearizing with the stable inward fit result
      o2::track::TrackParCov saveInw{temporaryTrack};
      linRef = saveInw; // use refitted track as lin.reference
      float saveChi2 = temporaryTrack.getChi2();
      temporaryTrack.resetCovariance();
      temporaryTrack.setCov(temporaryTrack.getQ2Pt() * temporaryTrack.getQ2Pt() * temporaryTrack.getCov()[o2::track::CovLabels::kSigQ2Pt2], o2::track::CovLabels::kSigQ2Pt2);
      temporaryTrack.setChi2(0);
      fitSuccess = fitTrack(temporaryTrack,               // TrackITSExt& track,
                            0,                            // int lastLayer,
                            NLayers,                      // int firstLayer,
                            1,                            // int firstCluster,
                            maxChi2ClusterAttachment,     // float maxChi2ClusterAttachment,
                            maxChi2NDF,                   // float maxChi2NDF,
                            o2::constants::math::VeryBig, // float maxQoverPt,
                            0,                            // nCl,
                            bz,                           // float bz,
                            foundTrackingFrameInfo,       // TrackingFrameInfo** trackingFrameInfo,
                            propagator,                   // const o2::base::Propagator* propagator,
                            matCorrType,                  // o2::base::PropagatorF::MatCorrType matCorrType
                            &linRef,
                            shifRefToCluster);
      if (!fitSuccess) {
        continue;
      }
      temporaryTrack.getParamOut() = temporaryTrack.getParamIn();
      temporaryTrack.getParamIn() = saveInw;
      temporaryTrack.setChi2(saveChi2);
    }

    if constexpr (initRun) {
      seedLUT[iCurrentTrackSeedIndex] = 1;
    } else {
      tracks[seedLUT[iCurrentTrackSeedIndex]] = temporaryTrack;
    }
  }
}

template <bool initRun, int NLayers = 7>
GPUg() void __launch_bounds__(256, 1) computeLayerCellNeighboursKernel(
  CellSeed<NLayers>** cellSeedArray,
  int* neighboursLUT,
  int* neighboursIndexTable,
  int** cellsLUTs,
  gpuPair<int, int>* cellNeighbours,
  const Tracklet** tracklets,
  const float maxChi2ClusterAttachment,
  const float bz,
  const int layerIndex,
  const unsigned int nCells,
  const int maxCellNeighbours = 1e2)
{
  for (int iCurrentCellIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCellIndex < nCells; iCurrentCellIndex += blockDim.x * gridDim.x) {
    if constexpr (!initRun) {
      if (neighboursIndexTable[iCurrentCellIndex] == neighboursIndexTable[iCurrentCellIndex + 1]) {
        continue;
      }
    }
    const auto& currentCellSeed{cellSeedArray[layerIndex][iCurrentCellIndex]};
    const int nextLayerTrackletIndex{currentCellSeed.getSecondTrackletIndex()};
    const int nextLayerFirstCellIndex{cellsLUTs[layerIndex + 1][nextLayerTrackletIndex]};
    const int nextLayerLastCellIndex{cellsLUTs[layerIndex + 1][nextLayerTrackletIndex + 1]};
    int foundNeighbours{0};
    for (int iNextCell{nextLayerFirstCellIndex}; iNextCell < nextLayerLastCellIndex; ++iNextCell) {
      auto nextCellSeed{cellSeedArray[layerIndex + 1][iNextCell]}; // Copy
      if (nextCellSeed.getFirstTrackletIndex() != nextLayerTrackletIndex || !currentCellSeed.getTimeStamp().isCompatible(nextCellSeed.getTimeStamp())) {
        break;
      }

      if (!nextCellSeed.rotate(currentCellSeed.getAlpha()) ||
          !nextCellSeed.propagateTo(currentCellSeed.getX(), bz)) {
        continue;
      }

      float chi2 = currentCellSeed.getPredictedChi2(nextCellSeed);
      if (chi2 > maxChi2ClusterAttachment) {
        continue;
      }

      if constexpr (initRun) {
        atomicAdd(neighboursLUT + iNextCell, 1);
        neighboursIndexTable[iCurrentCellIndex]++;
      } else {
        cellNeighbours[neighboursIndexTable[iCurrentCellIndex] + foundNeighbours] = {iCurrentCellIndex, iNextCell};
        foundNeighbours++;
        const int currentCellLevel{currentCellSeed.getLevel()};
        if (currentCellLevel >= nextCellSeed.getLevel()) {
          atomicMax(cellSeedArray[layerIndex + 1][iNextCell].getLevelPtr(), currentCellLevel + 1);
        }
      }
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(256, 1) computeLayerCellsKernel(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTrackletsCurrent,
  const int layer,
  CellSeed<NLayers>* cells,
  int** cellsLUTs,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut)
{
  constexpr float layerxX0[7] = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f}; // FIXME: Hardcoded here for the moment.
  for (int iCurrentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; iCurrentTrackletIndex < nTrackletsCurrent; iCurrentTrackletIndex += blockDim.x * gridDim.x) {
    if constexpr (!initRun) {
      if (cellsLUTs[layer][iCurrentTrackletIndex] == cellsLUTs[layer][iCurrentTrackletIndex + 1]) {
        continue;
      }
    }
    const Tracklet& currentTracklet = tracklets[layer][iCurrentTrackletIndex];
    const int nextLayerClusterIndex{currentTracklet.secondClusterIndex};
    const int nextLayerFirstTrackletIndex{trackletsLUT[layer + 1][nextLayerClusterIndex]};
    const int nextLayerLastTrackletIndex{trackletsLUT[layer + 1][nextLayerClusterIndex + 1]};
    if (nextLayerFirstTrackletIndex == nextLayerLastTrackletIndex) {
      continue;
    }
    int foundCells{0};
    for (int iNextTrackletIndex{nextLayerFirstTrackletIndex}; iNextTrackletIndex < nextLayerLastTrackletIndex; ++iNextTrackletIndex) {
      if (tracklets[layer + 1][iNextTrackletIndex].firstClusterIndex != nextLayerClusterIndex) {
        break;
      }
      const Tracklet& nextTracklet = tracklets[layer + 1][iNextTrackletIndex];
      if (!currentTracklet.getTimeStamp().isCompatible(nextTracklet.getTimeStamp())) {
        continue;
      }
      const float deltaTanLambda{o2::gpu::CAMath::Abs(currentTracklet.tanLambda - nextTracklet.tanLambda)};

      if (deltaTanLambda / cellDeltaTanLambdaSigma < nSigmaCut) {
        const int clusId[3]{
          sortedClusters[layer][currentTracklet.firstClusterIndex].clusterId,
          sortedClusters[layer + 1][nextTracklet.firstClusterIndex].clusterId,
          sortedClusters[layer + 2][nextTracklet.secondClusterIndex].clusterId};

        const auto& cluster1_glo = unsortedClusters[layer][clusId[0]];
        const auto& cluster2_glo = unsortedClusters[layer + 1][clusId[1]];
        const auto& cluster3_tf = tfInfo[layer + 2][clusId[2]];
        auto track{buildTrackSeed(cluster1_glo, cluster2_glo, cluster3_tf, bz)};
        float chi2{0.f};
        bool good{false};
        for (int iC{2}; iC--;) {
          const TrackingFrameInfo& trackingHit = tfInfo[layer + iC][clusId[iC]];
          if (!track.rotate(trackingHit.alphaTrackingFrame)) {
            break;
          }
          if (!track.propagateTo(trackingHit.xTrackingFrame, bz)) {
            break;
          }

          if (!track.correctForMaterial(layerxX0[layer + iC], layerxX0[layer + iC] * constants::Radl * constants::Rho, true)) {
            break;
          }

          const auto predChi2{track.getPredictedChi2Quiet(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)};
          if (!track.o2::track::TrackParCov::update(trackingHit.positionTrackingFrame, trackingHit.covarianceTrackingFrame)) {
            break;
          }
          if (!iC && predChi2 > maxChi2ClusterAttachment) {
            break;
          }
          good = !iC;
          chi2 += predChi2;
        }
        if (!good) {
          continue;
        }
        if constexpr (!initRun) {
          TimeEstBC ts = currentTracklet.getTimeStamp();
          ts += nextTracklet.getTimeStamp();
          new (cells + cellsLUTs[layer][iCurrentTrackletIndex] + foundCells) CellSeed<NLayers>{layer, clusId[0], clusId[1], clusId[2], iCurrentTrackletIndex, iNextTrackletIndex, track, chi2, ts};
        }
        ++foundCells;
      }
    }
    if constexpr (initRun) {
      cellsLUTs[layer][iCurrentTrackletIndex] = foundCells;
    }
  }
}

template <bool initRun, int NLayers>
GPUg() void __launch_bounds__(256, 1) computeLayerTrackletsMultiROFKernel(
  const IndexTableUtils<NLayers>* utils,
  const typename ROFMaskTable<NLayers>::View rofMask,
  const int layerIndex,
  const typename ROFOverlapTable<NLayers>::View rofOverlaps,
  const typename ROFVertexLookupTable<NLayers>::View vertexLUT,
  const Vertex* vertices,
  const int* rofPV,
  const int vertexId,
  const Cluster** clusters,
  const int** ROFClusters,
  const unsigned char** usedClusters,
  const int** indexTables,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int iteration,
  const float NSigmaCut,
  const float phiCut,
  const float resolutionPV,
  const float minR,
  const float maxR,
  const float positionResolution,
  const float meanDeltaR,
  const float MSAngle)
{
  const int phiBins{utils->getNphiBins()};
  const int zBins{utils->getNzBins()};
  const int tableSize{phiBins * zBins + 1};
  const int totalROFs0 = rofOverlaps.getLayer(layerIndex).mNROFsTF;
  const int totalROFs1 = rofOverlaps.getLayer(layerIndex + 1).mNROFsTF;
  for (unsigned int pivotROF{blockIdx.x}; pivotROF < totalROFs0; pivotROF += gridDim.x) {
    if (!rofMask.isROFEnabled(layerIndex, pivotROF)) {
      continue;
    }

    const auto& pvs = vertexLUT.getVertices(layerIndex, pivotROF);
    auto primaryVertices = gpuSpan<const Vertex>(&vertices[pvs.getFirstEntry()], pvs.getEntries());
    if (primaryVertices.empty()) {
      continue;
    }
    const auto startVtx{vertexId >= 0 ? vertexId : 0};
    const auto endVtx{vertexId >= 0 ? o2::gpu::CAMath::Min(vertexId + 1, static_cast<int>(primaryVertices.size())) : static_cast<int>(primaryVertices.size())};
    if (endVtx <= startVtx || (vertexId + 1) > primaryVertices.size()) {
      continue;
    }

    const auto& rofOverlap = rofOverlaps.getOverlap(layerIndex, layerIndex + 1, pivotROF);
    if (!rofOverlap.getEntries()) {
      continue;
    }

    auto clustersCurrentLayer = getClustersOnLayer(pivotROF, totalROFs0, layerIndex, ROFClusters, clusters);
    if (clustersCurrentLayer.empty()) {
      continue;
    }

    for (int currentClusterIndex = threadIdx.x; currentClusterIndex < clustersCurrentLayer.size(); currentClusterIndex += blockDim.x) {

      unsigned int storedTracklets{0};
      const auto& currentCluster{clustersCurrentLayer[currentClusterIndex]};
      const int currentSortedIndex{ROFClusters[layerIndex][pivotROF] + currentClusterIndex};
      if (usedClusters[layerIndex][currentCluster.clusterId]) {
        continue;
      }
      if constexpr (!initRun) {
        if (trackletsLUT[layerIndex][currentSortedIndex] == trackletsLUT[layerIndex][currentSortedIndex + 1]) {
          continue;
        }
      }

      const float inverseR0{1.f / currentCluster.radius};
      for (int iV{startVtx}; iV < endVtx; ++iV) {
        auto& primaryVertex{primaryVertices[iV]};
        if (!vertexLUT.isVertexCompatible(layerIndex, pivotROF, primaryVertex)) {
          continue;
        }
        if ((primaryVertex.isFlagSet(Vertex::Flags::UPCMode) && iteration != 3) || (iteration == 3 && !primaryVertex.isFlagSet(Vertex::Flags::UPCMode))) {
          continue;
        }

        const float resolution = o2::gpu::CAMath::Sqrt(math_utils::Sq(resolutionPV) / primaryVertex.getNContributors() + math_utils::Sq(positionResolution));
        const float tanLambda{(currentCluster.zCoordinate - primaryVertex.getZ()) * inverseR0};
        const float zAtRmin{tanLambda * (minR - currentCluster.radius) + currentCluster.zCoordinate};
        const float zAtRmax{tanLambda * (maxR - currentCluster.radius) + currentCluster.zCoordinate};
        const float sqInverseDeltaZ0{1.f / (math_utils::Sq(currentCluster.zCoordinate - primaryVertex.getZ()) + constants::Tolerance)}; /// protecting from overflows adding the detector resolution
        const float sigmaZ{o2::gpu::CAMath::Sqrt(math_utils::Sq(resolution) * math_utils::Sq(tanLambda) * ((math_utils::Sq(inverseR0) + sqInverseDeltaZ0) * math_utils::Sq(meanDeltaR) + 1.f) + math_utils::Sq(meanDeltaR * MSAngle))};
        const int4 selectedBinsRect{getBinsRect<NLayers>(currentCluster, layerIndex + 1, utils, zAtRmin, zAtRmax, sigmaZ * NSigmaCut, phiCut)};
        if (selectedBinsRect.x == 0 && selectedBinsRect.y == 0 && selectedBinsRect.z == 0 && selectedBinsRect.w == 0) {
          continue;
        }
        int phiBinsNum{selectedBinsRect.w - selectedBinsRect.y + 1};

        if (phiBinsNum < 0) {
          phiBinsNum += phiBins;
        }

        for (short targetROF = rofOverlap.getFirstEntry(); targetROF < rofOverlap.getEntriesBound(); ++targetROF) {
          if (!rofMask.isROFEnabled(layerIndex + 1, pivotROF)) {
            continue;
          }
          auto clustersNextLayer = getClustersOnLayer(targetROF, totalROFs1, layerIndex + 1, ROFClusters, clusters);
          if (clustersNextLayer.empty()) {
            continue;
          }
          const auto ts = rofOverlaps.getTimeStamp(layerIndex, pivotROF, layerIndex + 1, targetROF);
          if (!ts.isCompatible(primaryVertex.getTimeStamp())) {
            continue;
          }
          for (int iPhiCount{0}; iPhiCount < phiBinsNum; iPhiCount++) {
            int iPhiBin = (selectedBinsRect.y + iPhiCount) % phiBins;
            const int firstBinIndex{utils->getBinIndex(selectedBinsRect.x, iPhiBin)};
            const int maxBinIndex{firstBinIndex + selectedBinsRect.z - selectedBinsRect.x + 1};
            const int firstRowClusterIndex = indexTables[layerIndex + 1][(targetROF)*tableSize + firstBinIndex];
            const int maxRowClusterIndex = indexTables[layerIndex + 1][(targetROF)*tableSize + maxBinIndex];
            for (int nextClusterIndex{firstRowClusterIndex}; nextClusterIndex < maxRowClusterIndex; ++nextClusterIndex) {
              if (nextClusterIndex >= clustersNextLayer.size()) {
                break;
              }
              const Cluster& nextCluster{clustersNextLayer[nextClusterIndex]};
              if (usedClusters[layerIndex + 1][nextCluster.clusterId]) {
                continue;
              }
              const float deltaPhi{o2::gpu::CAMath::Abs(currentCluster.phi - nextCluster.phi)};
              const float deltaZ{o2::gpu::CAMath::Abs(tanLambda * (nextCluster.radius - currentCluster.radius) + currentCluster.zCoordinate - nextCluster.zCoordinate)};
              if (deltaZ / sigmaZ < NSigmaCut && (deltaPhi < phiCut || o2::gpu::CAMath::Abs(deltaPhi - o2::constants::math::TwoPI) < phiCut)) {
                if constexpr (initRun) {
                  trackletsLUT[layerIndex][currentSortedIndex]++; // we need l0 as well for usual exclusive sums.
                } else {
                  const float phi{o2::gpu::CAMath::ATan2(currentCluster.yCoordinate - nextCluster.yCoordinate, currentCluster.xCoordinate - nextCluster.xCoordinate)};
                  const float tanL{(currentCluster.zCoordinate - nextCluster.zCoordinate) / (currentCluster.radius - nextCluster.radius)};
                  const int nextSortedIndex{ROFClusters[layerIndex + 1][targetROF] + nextClusterIndex};
                  new (tracklets[layerIndex] + trackletsLUT[layerIndex][currentSortedIndex] + storedTracklets) Tracklet{currentSortedIndex, nextSortedIndex, tanL, phi, ts};
                }
                ++storedTracklets;
              }
            }
          }
        }
      }
    }
  }
}

GPUg() void __launch_bounds__(256, 1) compileTrackletsLookupTableKernel(
  const Tracklet* tracklets,
  int* trackletsLookUpTable,
  const int nTracklets)
{
  for (int currentTrackletIndex = blockIdx.x * blockDim.x + threadIdx.x; currentTrackletIndex < nTracklets; currentTrackletIndex += blockDim.x * gridDim.x) {
    atomicAdd(&trackletsLookUpTable[tracklets[currentTrackletIndex].firstClusterIndex], 1);
  }
}

template <bool dryRun, int NLayers = 7>
GPUg() void __launch_bounds__(256, 1) processNeighboursKernel(
  const int layer,
  const int level,
  CellSeed<NLayers>** allCellSeeds,
  CellSeed<NLayers>* currentCellSeeds,
  const int* currentCellIds,
  const unsigned int nCurrentCells,
  CellSeed<NLayers>* updatedCellSeeds,
  int* updatedCellsIds,
  int* foundSeedsTable,               // auxiliary only in GPU code to compute the number of cells per iteration
  const unsigned char** usedClusters, // Used clusters
  int* neighbours,
  int* neighboursLUT,
  const TrackingFrameInfo** foundTrackingFrameInfo,
  const float bz,
  const float maxChi2ClusterAttachment,
  const o2::base::Propagator* propagator,
  const o2::base::PropagatorF::MatCorrType matCorrType)
{
  constexpr float layerxX0[7] = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f}; // FIXME: Hardcoded here for the moment.
  for (unsigned int iCurrentCell = blockIdx.x * blockDim.x + threadIdx.x; iCurrentCell < nCurrentCells; iCurrentCell += blockDim.x * gridDim.x) {
    if constexpr (!dryRun) {
      if (foundSeedsTable[iCurrentCell] == foundSeedsTable[iCurrentCell + 1]) {
        continue;
      }
    }
    int foundSeeds{0};
    const auto& currentCell{currentCellSeeds[iCurrentCell]};
    if (currentCell.getLevel() != level) {
      continue;
    }
    if (currentCellIds == nullptr && (usedClusters[layer][currentCell.getFirstClusterIndex()] ||
                                      usedClusters[layer + 1][currentCell.getSecondClusterIndex()] ||
                                      usedClusters[layer + 2][currentCell.getThirdClusterIndex()])) {
      continue;
    }
    const int cellId = currentCellIds == nullptr ? iCurrentCell : currentCellIds[iCurrentCell];

    const int startNeighbourId{cellId ? neighboursLUT[cellId - 1] : 0};
    const int endNeighbourId{neighboursLUT[cellId]};

    for (int iNeighbourCell{startNeighbourId}; iNeighbourCell < endNeighbourId; ++iNeighbourCell) {
      const int neighbourCellId = neighbours[iNeighbourCell];
      const auto& neighbourCell = allCellSeeds[layer - 1][neighbourCellId];

      if (neighbourCell.getSecondTrackletIndex() != currentCell.getFirstTrackletIndex()) {
        continue;
      }
      if (!currentCell.getTimeStamp().isCompatible(neighbourCell.getTimeStamp())) {
        continue;
      }
      if (currentCell.getLevel() - 1 != neighbourCell.getLevel()) {
        continue;
      }
      if (usedClusters[layer - 1][neighbourCell.getFirstClusterIndex()]) {
        continue;
      }
      auto seed{currentCell};
      auto& trHit = foundTrackingFrameInfo[layer - 1][neighbourCell.getFirstClusterIndex()];

      if (!seed.rotate(trHit.alphaTrackingFrame)) {
        continue;
      }

      if (!propagator->propagateToX(seed, trHit.xTrackingFrame, bz, o2::base::PropagatorImpl<float>::MAX_SIN_PHI, o2::base::PropagatorImpl<float>::MAX_STEP, matCorrType)) {
        continue;
      }

      if (matCorrType == o2::base::PropagatorF::MatCorrType::USEMatCorrNONE) {
        if (!seed.correctForMaterial(layerxX0[layer - 1], layerxX0[layer - 1] * constants::Radl * constants::Rho, true)) {
          continue;
        }
      }

      auto predChi2{seed.getPredictedChi2Quiet(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)};
      if ((predChi2 > maxChi2ClusterAttachment) || predChi2 < 0.f) {
        continue;
      }
      seed.setChi2(seed.getChi2() + predChi2);
      if (!seed.o2::track::TrackParCov::update(trHit.positionTrackingFrame, trHit.covarianceTrackingFrame)) {
        continue;
      }
      if constexpr (dryRun) {
        foundSeedsTable[iCurrentCell]++;
      } else {
        seed.getClusters()[layer - 1] = neighbourCell.getFirstClusterIndex();
        seed.setLevel(neighbourCell.getLevel());
        seed.setFirstTrackletIndex(neighbourCell.getFirstTrackletIndex());
        seed.setSecondTrackletIndex(neighbourCell.getSecondTrackletIndex());
        updatedCellsIds[foundSeedsTable[iCurrentCell] + foundSeeds] = neighbourCellId;
        updatedCellSeeds[foundSeedsTable[iCurrentCell] + foundSeeds] = seed;
      }
      foundSeeds++;
    }
  }
}

} // namespace gpu

template <int NLayers>
void countTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                 const typename ROFMaskTable<NLayers>::View& rofMask,
                                 const int layer,
                                 const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                 const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                 const int vertexId,
                                 const Vertex* vertices,
                                 const int* rofPV,
                                 const Cluster** clusters,
                                 std::vector<unsigned int> nClusters,
                                 const int** ROFClusters,
                                 const unsigned char** usedClusters,
                                 const int** clustersIndexTables,
                                 int** trackletsLUTs,
                                 gsl::span<int*> trackletsLUTsHost,
                                 const int iteration,
                                 const float NSigmaCut,
                                 bounded_vector<float>& phiCuts,
                                 const float resolutionPV,
                                 std::array<float, NLayers>& minRs,
                                 std::array<float, NLayers>& maxRs,
                                 bounded_vector<float>& resolutions,
                                 std::vector<float>& radii,
                                 bounded_vector<float>& mulScatAng,
                                 o2::its::ExternalAllocator* alloc,
                                 const int nBlocks,
                                 const int nThreads,
                                 gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<true><<<nBlocks, nThreads, 0, streams[layer].get()>>>(
    utils,
    rofMask,
    layer,
    rofOverlaps,
    vertexLUT,
    vertices,
    rofPV,
    vertexId,
    clusters,
    ROFClusters,
    usedClusters,
    clustersIndexTables,
    nullptr,
    trackletsLUTs,
    iteration,
    NSigmaCut,
    phiCuts[layer],
    resolutionPV,
    minRs[layer + 1],
    maxRs[layer + 1],
    resolutions[layer],
    radii[layer + 1] - radii[layer],
    mulScatAng[layer]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[layer].get());
  thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[layer], trackletsLUTsHost[layer] + nClusters[layer] + 1, trackletsLUTsHost[layer]);
}

template <int NLayers>
void computeTrackletsInROFsHandler(const IndexTableUtils<NLayers>* utils,
                                   const typename ROFMaskTable<NLayers>::View& rofMask,
                                   const int layer,
                                   const typename ROFOverlapTable<NLayers>::View& rofOverlaps,
                                   const typename ROFVertexLookupTable<NLayers>::View& vertexLUT,
                                   const int vertexId,
                                   const Vertex* vertices,
                                   const int* rofPV,
                                   const Cluster** clusters,
                                   std::vector<unsigned int> nClusters,
                                   const int** ROFClusters,
                                   const unsigned char** usedClusters,
                                   const int** clustersIndexTables,
                                   Tracklet** tracklets,
                                   gsl::span<Tracklet*> spanTracklets,
                                   gsl::span<int> nTracklets,
                                   int** trackletsLUTs,
                                   gsl::span<int*> trackletsLUTsHost,
                                   const int iteration,
                                   const float NSigmaCut,
                                   bounded_vector<float>& phiCuts,
                                   const float resolutionPV,
                                   std::array<float, NLayers>& minRs,
                                   std::array<float, NLayers>& maxRs,
                                   bounded_vector<float>& resolutions,
                                   std::vector<float>& radii,
                                   bounded_vector<float>& mulScatAng,
                                   o2::its::ExternalAllocator* alloc,
                                   const int nBlocks,
                                   const int nThreads,
                                   gpu::Streams& streams)
{
  gpu::computeLayerTrackletsMultiROFKernel<false><<<nBlocks, nThreads, 0, streams[layer].get()>>>(
    utils,
    rofMask,
    layer,
    rofOverlaps,
    vertexLUT,
    vertices,
    rofPV,
    vertexId,
    clusters,
    ROFClusters,
    usedClusters,
    clustersIndexTables,
    tracklets,
    trackletsLUTs,
    iteration,
    NSigmaCut,
    phiCuts[layer],
    resolutionPV,
    minRs[layer + 1],
    maxRs[layer + 1],
    resolutions[layer],
    radii[layer + 1] - radii[layer],
    mulScatAng[layer]);
  thrust::device_ptr<Tracklet> tracklets_ptr(spanTracklets[layer]);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[layer].get());
  thrust::sort(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[layer], gpu::sort_tracklets());
  auto unique_end = thrust::unique(nosync_policy, tracklets_ptr, tracklets_ptr + nTracklets[layer], gpu::equal_tracklets());
  nTracklets[layer] = unique_end - tracklets_ptr;
  if (layer) {
    GPUChkErrS(cudaMemsetAsync(trackletsLUTsHost[layer], 0, (nClusters[layer] + 1) * sizeof(int), streams[layer].get()));
    gpu::compileTrackletsLookupTableKernel<<<nBlocks, nThreads, 0, streams[layer].get()>>>(
      spanTracklets[layer],
      trackletsLUTsHost[layer],
      nTracklets[layer]);
    thrust::exclusive_scan(nosync_policy, trackletsLUTsHost[layer], trackletsLUTsHost[layer] + nClusters[layer] + 1, trackletsLUTsHost[layer]);
  }
}

template <int NLayers>
void countCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int layer,
  CellSeed<NLayers>* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  o2::its::ExternalAllocator* alloc,
  const int nBlocks,
  const int nThreads,
  gpu::Streams& streams)
{
  gpu::computeLayerCellsKernel<true><<<nBlocks, nThreads, 0, streams[layer].get()>>>(
    sortedClusters,           // const Cluster**
    unsortedClusters,         // const Cluster**
    tfInfo,                   // const TrackingFrameInfo**
    tracklets,                // const Tracklets**
    trackletsLUT,             // const int**
    nTracklets,               // const int
    layer,                    // const int
    cells,                    // CellSeed*
    cellsLUTsArrayDevice,     // int**
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(streams[layer].get());
  thrust::exclusive_scan(nosync_policy, cellsLUTsHost, cellsLUTsHost + nTracklets + 1, cellsLUTsHost);
}

template <int NLayers>
void computeCellsHandler(
  const Cluster** sortedClusters,
  const Cluster** unsortedClusters,
  const TrackingFrameInfo** tfInfo,
  Tracklet** tracklets,
  int** trackletsLUT,
  const int nTracklets,
  const int layer,
  CellSeed<NLayers>* cells,
  int** cellsLUTsArrayDevice,
  int* cellsLUTsHost,
  const float bz,
  const float maxChi2ClusterAttachment,
  const float cellDeltaTanLambdaSigma,
  const float nSigmaCut,
  const int nBlocks,
  const int nThreads,
  gpu::Streams& streams)
{
  gpu::computeLayerCellsKernel<false><<<nBlocks, nThreads, 0, streams[layer].get()>>>(
    sortedClusters,           // const Cluster**
    unsortedClusters,         // const Cluster**
    tfInfo,                   // const TrackingFrameInfo**
    tracklets,                // const Tracklets**
    trackletsLUT,             // const int**
    nTracklets,               // const int
    layer,                    // const int
    cells,                    // CellSeed*
    cellsLUTsArrayDevice,     // int**
    bz,                       // const float
    maxChi2ClusterAttachment, // const float
    cellDeltaTanLambdaSigma,  // const float
    nSigmaCut);               // const float
}

template <int NLayers>
void countCellNeighboursHandler(CellSeed<NLayers>** cellsLayersDevice,
                                int* neighboursLUT,
                                int** cellsLUTs,
                                gpuPair<int, int>* cellNeighbours,
                                int* neighboursIndexTable,
                                const Tracklet** tracklets,
                                const float maxChi2ClusterAttachment,
                                const float bz,
                                const int layerIndex,
                                const unsigned int nCells,
                                const unsigned int nCellsNext,
                                const int maxCellNeighbours,
                                o2::its::ExternalAllocator* alloc,
                                const int nBlocks,
                                const int nThreads,
                                gpu::Stream& stream)
{
  gpu::computeLayerCellNeighboursKernel<true><<<nBlocks, nThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    neighboursLUT,
    neighboursIndexTable,
    cellsLUTs,
    cellNeighbours,
    tracklets,
    maxChi2ClusterAttachment,
    bz,
    layerIndex,
    nCells,
    maxCellNeighbours);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(stream.get());
  thrust::inclusive_scan(nosync_policy, neighboursLUT, neighboursLUT + nCellsNext, neighboursLUT);
  thrust::exclusive_scan(nosync_policy, neighboursIndexTable, neighboursIndexTable + nCells + 1, neighboursIndexTable);
}

template <int NLayers>
void computeCellNeighboursHandler(CellSeed<NLayers>** cellsLayersDevice,
                                  int* neighboursLUT,
                                  int** cellsLUTs,
                                  gpuPair<int, int>* cellNeighbours,
                                  int* neighboursIndexTable,
                                  const Tracklet** tracklets,
                                  const float maxChi2ClusterAttachment,
                                  const float bz,
                                  const int layerIndex,
                                  const unsigned int nCells,
                                  const unsigned int nCellsNext,
                                  const int maxCellNeighbours,
                                  const int nBlocks,
                                  const int nThreads,
                                  gpu::Stream& stream)
{
  gpu::computeLayerCellNeighboursKernel<false><<<nBlocks, nThreads, 0, stream.get()>>>(
    cellsLayersDevice,
    neighboursLUT,
    neighboursIndexTable,
    cellsLUTs,
    cellNeighbours,
    tracklets,
    maxChi2ClusterAttachment,
    bz,
    layerIndex,
    nCells,
    maxCellNeighbours);
}

int filterCellNeighboursHandler(gpuPair<int, int>* cellNeighbourPairs,
                                int* cellNeighbours,
                                unsigned int nNeigh,
                                gpu::Stream& stream,
                                o2::its::ExternalAllocator* allocator)
{
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(allocator)).on(stream.get());
  thrust::device_ptr<gpuPair<int, int>> neighVectorPairs(cellNeighbourPairs);
  thrust::device_ptr<int> validNeighs(cellNeighbours);
  auto updatedEnd = thrust::remove_if(nosync_policy, neighVectorPairs, neighVectorPairs + nNeigh, gpu::is_invalid_pair<int, int>());
  size_t newSize = updatedEnd - neighVectorPairs;
  thrust::stable_sort(nosync_policy, neighVectorPairs, neighVectorPairs + newSize, gpu::sort_by_second<int, int>());
  thrust::transform(nosync_policy, neighVectorPairs, neighVectorPairs + newSize, validNeighs, gpu::pair_to_first<int, int>());
  return newSize;
}

template <int NLayers>
void processNeighboursHandler(const int startLayer,
                              const int startLevel,
                              CellSeed<NLayers>** allCellSeeds,
                              CellSeed<NLayers>* currentCellSeeds,
                              std::array<int, NLayers - 2>& nCells,
                              const unsigned char** usedClusters,
                              std::array<int*, NLayers - 2>& neighbours,
                              gsl::span<int*> neighboursDeviceLUTs,
                              const TrackingFrameInfo** foundTrackingFrameInfo,
                              bounded_vector<CellSeed<NLayers>>& seedsHost,
                              const float bz,
                              const float maxChi2ClusterAttachment,
                              const float maxChi2NDF,
                              const o2::base::Propagator* propagator,
                              const o2::base::PropagatorF::MatCorrType matCorrType,
                              o2::its::ExternalAllocator* alloc,
                              const int nBlocks,
                              const int nThreads)
{
  constexpr uint64_t Tag = qStr2Tag("ITS_PNH1");
  alloc->pushTagOnStack(Tag);
  auto allocInt = gpu::TypedAllocator<int>(alloc);
  auto allocCellSeed = gpu::TypedAllocator<CellSeed<NLayers>>(alloc);
  thrust::device_vector<int, gpu::TypedAllocator<int>> foundSeedsTable(nCells[startLayer] + 1, 0, allocInt);
  auto nosync_policy = THRUST_NAMESPACE::par_nosync(gpu::TypedAllocator<char>(alloc)).on(gpu::Stream::DefaultStream);

  gpu::processNeighboursKernel<true, NLayers><<<nBlocks, nThreads>>>(
    startLayer,
    startLevel,
    allCellSeeds,
    currentCellSeeds,
    nullptr,
    nCells[startLayer],
    nullptr,
    nullptr,
    thrust::raw_pointer_cast(&foundSeedsTable[0]),
    usedClusters,
    neighbours[startLayer - 1],
    neighboursDeviceLUTs[startLayer - 1],
    foundTrackingFrameInfo,
    bz,
    maxChi2ClusterAttachment,
    propagator,
    matCorrType);
  thrust::exclusive_scan(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), foundSeedsTable.begin());

  thrust::device_vector<int, gpu::TypedAllocator<int>> updatedCellId(foundSeedsTable.back(), 0, allocInt);
  thrust::device_vector<CellSeed<NLayers>, gpu::TypedAllocator<CellSeed<NLayers>>> updatedCellSeed(foundSeedsTable.back(), allocCellSeed);
  gpu::processNeighboursKernel<false, NLayers><<<nBlocks, nThreads>>>(
    startLayer,
    startLevel,
    allCellSeeds,
    currentCellSeeds,
    nullptr,
    nCells[startLayer],
    thrust::raw_pointer_cast(&updatedCellSeed[0]),
    thrust::raw_pointer_cast(&updatedCellId[0]),
    thrust::raw_pointer_cast(&foundSeedsTable[0]),
    usedClusters,
    neighbours[startLayer - 1],
    neighboursDeviceLUTs[startLayer - 1],
    foundTrackingFrameInfo,
    bz,
    maxChi2ClusterAttachment,
    propagator,
    matCorrType);
  GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));

  int level = startLevel;
  thrust::device_vector<int, gpu::TypedAllocator<int>> lastCellId(allocInt);
  thrust::device_vector<CellSeed<NLayers>, gpu::TypedAllocator<CellSeed<NLayers>>> lastCellSeed(allocCellSeed);
  for (int iLayer{startLayer - 1}; iLayer > 0 && level > 2; --iLayer) {
    lastCellSeed.swap(updatedCellSeed);
    lastCellId.swap(updatedCellId);
    thrust::device_vector<CellSeed<NLayers>, gpu::TypedAllocator<CellSeed<NLayers>>>(allocCellSeed).swap(updatedCellSeed);
    thrust::device_vector<int, gpu::TypedAllocator<int>>(allocInt).swap(updatedCellId);
    auto lastCellSeedSize{lastCellSeed.size()};
    foundSeedsTable.resize(lastCellSeedSize + 1);
    thrust::fill(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), 0);

    gpu::processNeighboursKernel<true, NLayers><<<nBlocks, nThreads>>>(
      iLayer,
      --level,
      allCellSeeds,
      thrust::raw_pointer_cast(&lastCellSeed[0]),
      thrust::raw_pointer_cast(&lastCellId[0]),
      lastCellSeedSize,
      nullptr,
      nullptr,
      thrust::raw_pointer_cast(&foundSeedsTable[0]),
      usedClusters,
      neighbours[iLayer - 1],
      neighboursDeviceLUTs[iLayer - 1],
      foundTrackingFrameInfo,
      bz,
      maxChi2ClusterAttachment,
      propagator,
      matCorrType);
    thrust::exclusive_scan(nosync_policy, foundSeedsTable.begin(), foundSeedsTable.end(), foundSeedsTable.begin());

    auto foundSeeds{foundSeedsTable.back()};
    updatedCellId.resize(foundSeeds);
    thrust::fill(nosync_policy, updatedCellId.begin(), updatedCellId.end(), 0);
    updatedCellSeed.resize(foundSeeds);
    thrust::fill(nosync_policy, updatedCellSeed.begin(), updatedCellSeed.end(), CellSeed<NLayers>());

    gpu::processNeighboursKernel<false, NLayers><<<nBlocks, nThreads>>>(
      iLayer,
      level,
      allCellSeeds,
      thrust::raw_pointer_cast(&lastCellSeed[0]),
      thrust::raw_pointer_cast(&lastCellId[0]),
      lastCellSeedSize,
      thrust::raw_pointer_cast(&updatedCellSeed[0]),
      thrust::raw_pointer_cast(&updatedCellId[0]),
      thrust::raw_pointer_cast(&foundSeedsTable[0]),
      usedClusters,
      neighbours[iLayer - 1],
      neighboursDeviceLUTs[iLayer - 1],
      foundTrackingFrameInfo,
      bz,
      maxChi2ClusterAttachment,
      propagator,
      matCorrType);
  }
  GPUChkErrS(cudaStreamSynchronize(gpu::Stream::DefaultStream));
  thrust::device_vector<CellSeed<NLayers>, gpu::TypedAllocator<CellSeed<NLayers>>> outSeeds(updatedCellSeed.size(), allocCellSeed);
  auto end = thrust::copy_if(nosync_policy, updatedCellSeed.begin(), updatedCellSeed.end(), outSeeds.begin(), gpu::seed_selector<NLayers>(1.e3, maxChi2NDF * ((startLevel + 2) * 2 - 5)));
  auto s{end - outSeeds.begin()};
  seedsHost.reserve(seedsHost.size() + s);
  thrust::copy(outSeeds.begin(), outSeeds.begin() + s, std::back_inserter(seedsHost));
  alloc->popTagOffStack(Tag);
}

template <int NLayers>
void countTrackSeedHandler(CellSeed<NLayers>* trackSeeds,
                           const TrackingFrameInfo** foundTrackingFrameInfo,
                           const Cluster** unsortedClusters,
                           int* seedLUT,
                           const std::vector<float>& layerRadiiHost,
                           const std::vector<float>& minPtsHost,
                           const unsigned int nSeeds,
                           const float bz,
                           const int startLevel,
                           const float maxChi2ClusterAttachment,
                           const float maxChi2NDF,
                           const int reseedIfShorter,
                           const bool repeatRefitOut,
                           const bool shiftRefToCluster,
                           const o2::base::Propagator* propagator,
                           const o2::base::PropagatorF::MatCorrType matCorrType,
                           o2::its::ExternalAllocator* alloc,
                           const int nBlocks,
                           const int nThreads)
{
  // TODO: the minPts&layerRadii is transfered twice
  // we should allocate this in constant memory and stop these
  // small transferes!
  thrust::device_vector<float> minPts(minPtsHost);
  thrust::device_vector<float> layerRadii(layerRadiiHost);
  gpu::fitTrackSeedsKernel<true, NLayers><<<nBlocks, nThreads>>>(
    trackSeeds,                               // CellSeed*
    foundTrackingFrameInfo,                   // TrackingFrameInfo**
    unsortedClusters,                         // Cluster**
    nullptr,                                  // TrackITSExt*
    seedLUT,                                  // int*
    thrust::raw_pointer_cast(&layerRadii[0]), // const float*
    thrust::raw_pointer_cast(&minPts[0]),     // const float*
    nSeeds,                                   // const unsigned int
    bz,                                       // const float
    startLevel,                               // const int
    maxChi2ClusterAttachment,                 // float
    maxChi2NDF,                               // float
    reseedIfShorter,                          // int
    repeatRefitOut,                           // bool
    shiftRefToCluster,                        // bool
    propagator,                               // const o2::base::Propagator*
    matCorrType);                             // o2::base::PropagatorF::MatCorrType
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::exclusive_scan(sync_policy, seedLUT, seedLUT + nSeeds + 1, seedLUT);
}

template <int NLayers>
void computeTrackSeedHandler(CellSeed<NLayers>* trackSeeds,
                             const TrackingFrameInfo** foundTrackingFrameInfo,
                             const Cluster** unsortedClusters,
                             o2::its::TrackITSExt* tracks,
                             const int* seedLUT,
                             const std::vector<float>& layerRadiiHost,
                             const std::vector<float>& minPtsHost,
                             const unsigned int nSeeds,
                             const unsigned int nTracks,
                             const float bz,
                             const int startLevel,
                             const float maxChi2ClusterAttachment,
                             const float maxChi2NDF,
                             const int reseedIfShorter,
                             const bool repeatRefitOut,
                             const bool shiftRefToCluster,
                             const o2::base::Propagator* propagator,
                             const o2::base::PropagatorF::MatCorrType matCorrType,
                             o2::its::ExternalAllocator* alloc,
                             const int nBlocks,
                             const int nThreads)
{
  thrust::device_vector<float> minPts(minPtsHost);
  thrust::device_vector<float> layerRadii(layerRadiiHost);
  gpu::fitTrackSeedsKernel<false, NLayers><<<nBlocks, nThreads>>>(
    trackSeeds,                               // CellSeed*
    foundTrackingFrameInfo,                   // TrackingFrameInfo**
    unsortedClusters,                         // Cluster**
    tracks,                                   // TrackITSExt*
    seedLUT,                                  // const int*
    thrust::raw_pointer_cast(&layerRadii[0]), // const float*
    thrust::raw_pointer_cast(&minPts[0]),     // const float*
    nSeeds,                                   // const unsigned int
    bz,                                       // const float
    startLevel,                               // const int
    maxChi2ClusterAttachment,                 // float
    maxChi2NDF,                               // float
    reseedIfShorter,                          // int
    repeatRefitOut,                           // bool
    shiftRefToCluster,                        // bool
    propagator,                               // const o2::base::Propagator*
    matCorrType);                             // o2::base::PropagatorF::MatCorrType
  auto sync_policy = THRUST_NAMESPACE::par(gpu::TypedAllocator<char>(alloc));
  thrust::device_ptr<o2::its::TrackITSExt> tr_ptr(tracks);
  thrust::sort(sync_policy, tr_ptr, tr_ptr + nTracks, gpu::compare_track_chi2());
}

/// Explicit instantiation of ITS2 handlers
template void countTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                             const ROFMaskTable<7>::View& rofMask,
                                             const int layer,
                                             const ROFOverlapTable<7>::View& rofOverlaps,
                                             const ROFVertexLookupTable<7>::View& vertexLUT,
                                             const int vertexId,
                                             const Vertex* vertices,
                                             const int* rofPV,
                                             const Cluster** clusters,
                                             std::vector<unsigned int> nClusters,
                                             const int** ROFClusters,
                                             const unsigned char** usedClusters,
                                             const int** clustersIndexTables,
                                             int** trackletsLUTs,
                                             gsl::span<int*> trackletsLUTsHost,
                                             const int iteration,
                                             const float NSigmaCut,
                                             bounded_vector<float>& phiCuts,
                                             const float resolutionPV,
                                             std::array<float, 7>& minRs,
                                             std::array<float, 7>& maxRs,
                                             bounded_vector<float>& resolutions,
                                             std::vector<float>& radii,
                                             bounded_vector<float>& mulScatAng,
                                             o2::its::ExternalAllocator* alloc,
                                             const int nBlocks,
                                             const int nThreads,
                                             gpu::Streams& streams);

template void computeTrackletsInROFsHandler<7>(const IndexTableUtils<7>* utils,
                                               const ROFMaskTable<7>::View& rofMask,
                                               const int layer,
                                               const ROFOverlapTable<7>::View& rofOverlaps,
                                               const ROFVertexLookupTable<7>::View& vertexLUT,
                                               const int vertexId,
                                               const Vertex* vertices,
                                               const int* rofPV,
                                               const Cluster** clusters,
                                               std::vector<unsigned int> nClusters,
                                               const int** ROFClusters,
                                               const unsigned char** usedClusters,
                                               const int** clustersIndexTables,
                                               Tracklet** tracklets,
                                               gsl::span<Tracklet*> spanTracklets,
                                               gsl::span<int> nTracklets,
                                               int** trackletsLUTs,
                                               gsl::span<int*> trackletsLUTsHost,
                                               const int iteration,
                                               const float NSigmaCut,
                                               bounded_vector<float>& phiCuts,
                                               const float resolutionPV,
                                               std::array<float, 7>& minRs,
                                               std::array<float, 7>& maxRs,
                                               bounded_vector<float>& resolutions,
                                               std::vector<float>& radii,
                                               bounded_vector<float>& mulScatAng,
                                               o2::its::ExternalAllocator* alloc,
                                               const int nBlocks,
                                               const int nThreads,
                                               gpu::Streams& streams);

template void countCellsHandler<7>(const Cluster** sortedClusters,
                                   const Cluster** unsortedClusters,
                                   const TrackingFrameInfo** tfInfo,
                                   Tracklet** tracklets,
                                   int** trackletsLUT,
                                   const int nTracklets,
                                   const int layer,
                                   CellSeed<7>* cells,
                                   int** cellsLUTsArrayDevice,
                                   int* cellsLUTsHost,
                                   const float bz,
                                   const float maxChi2ClusterAttachment,
                                   const float cellDeltaTanLambdaSigma,
                                   const float nSigmaCut,
                                   o2::its::ExternalAllocator* alloc,
                                   const int nBlocks,
                                   const int nThreads,
                                   gpu::Streams& streams);

template void computeCellsHandler<7>(const Cluster** sortedClusters,
                                     const Cluster** unsortedClusters,
                                     const TrackingFrameInfo** tfInfo,
                                     Tracklet** tracklets,
                                     int** trackletsLUT,
                                     const int nTracklets,
                                     const int layer,
                                     CellSeed<7>* cells,
                                     int** cellsLUTsArrayDevice,
                                     int* cellsLUTsHost,
                                     const float bz,
                                     const float maxChi2ClusterAttachment,
                                     const float cellDeltaTanLambdaSigma,
                                     const float nSigmaCut,
                                     const int nBlocks,
                                     const int nThreads,
                                     gpu::Streams& streams);

template void countCellNeighboursHandler<7>(CellSeed<7>** cellsLayersDevice,
                                            int* neighboursLUT,
                                            int** cellsLUTs,
                                            gpuPair<int, int>* cellNeighbours,
                                            int* neighboursIndexTable,
                                            const Tracklet** tracklets,
                                            const float maxChi2ClusterAttachment,
                                            const float bz,
                                            const int layerIndex,
                                            const unsigned int nCells,
                                            const unsigned int nCellsNext,
                                            const int maxCellNeighbours,
                                            o2::its::ExternalAllocator* alloc,
                                            const int nBlocks,
                                            const int nThreads,
                                            gpu::Stream& stream);

template void computeCellNeighboursHandler(CellSeed<7>** cellsLayersDevice,
                                           int* neighboursLUT,
                                           int** cellsLUTs,
                                           gpuPair<int, int>* cellNeighbours,
                                           int* neighboursIndexTable,
                                           const Tracklet** tracklets,
                                           const float maxChi2ClusterAttachment,
                                           const float bz,
                                           const int layerIndex,
                                           const unsigned int nCells,
                                           const unsigned int nCellsNext,
                                           const int maxCellNeighbours,
                                           const int nBlocks,
                                           const int nThreads,
                                           gpu::Stream& stream);

template void processNeighboursHandler<7>(const int startLayer,
                                          const int startLevel,
                                          CellSeed<7>** allCellSeeds,
                                          CellSeed<7>* currentCellSeeds,
                                          std::array<int, 5>& nCells,
                                          const unsigned char** usedClusters,
                                          std::array<int*, 5>& neighbours,
                                          gsl::span<int*> neighboursDeviceLUTs,
                                          const TrackingFrameInfo** foundTrackingFrameInfo,
                                          bounded_vector<CellSeed<7>>& seedsHost,
                                          const float bz,
                                          const float maxChi2ClusterAttachment,
                                          const float maxChi2NDF,
                                          const o2::base::Propagator* propagator,
                                          const o2::base::PropagatorF::MatCorrType matCorrType,
                                          o2::its::ExternalAllocator* alloc,
                                          const int nBlocks,
                                          const int nThreads);

template void countTrackSeedHandler(CellSeed<7>* trackSeeds,
                                    const TrackingFrameInfo** foundTrackingFrameInfo,
                                    const Cluster** unsortedClusters,
                                    int* seedLUT,
                                    const std::vector<float>& layerRadiiHost,
                                    const std::vector<float>& minPtsHost,
                                    const unsigned int nSeeds,
                                    const float bz,
                                    const int startLevel,
                                    const float maxChi2ClusterAttachment,
                                    const float maxChi2NDF,
                                    const int reseedIfShorter,
                                    const bool repeatRefitOut,
                                    const bool shiftRefToCluster,
                                    const o2::base::Propagator* propagator,
                                    const o2::base::PropagatorF::MatCorrType matCorrType,
                                    o2::its::ExternalAllocator* alloc,
                                    const int nBlocks,
                                    const int nThreads);

template void computeTrackSeedHandler(CellSeed<7>* trackSeeds,
                                      const TrackingFrameInfo** foundTrackingFrameInfo,
                                      const Cluster** unsortedClusters,
                                      o2::its::TrackITSExt* tracks,
                                      const int* seedLUT,
                                      const std::vector<float>& layerRadiiHost,
                                      const std::vector<float>& minPtsHost,
                                      const unsigned int nSeeds,
                                      const unsigned int nTracks,
                                      const float bz,
                                      const int startLevel,
                                      const float maxChi2ClusterAttachment,
                                      const float maxChi2NDF,
                                      const int reseedIfShorter,
                                      const bool repeatRefitOut,
                                      const bool shiftRefToCluster,
                                      const o2::base::Propagator* propagator,
                                      const o2::base::PropagatorF::MatCorrType matCorrType,
                                      o2::its::ExternalAllocator* alloc,
                                      const int nBlocks,
                                      const int nThreads);

} // namespace o2::its
