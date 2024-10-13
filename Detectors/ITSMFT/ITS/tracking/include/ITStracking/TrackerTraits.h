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
/// \file TrackerTraits.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_
#define TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <functional>

#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/MatLayerCylSet.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Road.h"

// #define OPTIMISATION_OUTPUT

namespace o2
{
namespace gpu
{
class GPUChainITS;
}
namespace its
{
class TrackITSExt;

class TrackerTraits
{
 public:
  virtual ~TrackerTraits() = default;
  virtual void adoptTimeFrame(TimeFrame* tf);
  virtual void initialiseTimeFrame(const int iteration);
  virtual void computeLayerTracklets(const int iteration, int iROFslice, int iVertex);
  virtual void computeLayerCells(const int iteration);
  virtual void findCellsNeighbours(const int iteration);
  virtual void findRoads(const int iteration);
  virtual void initialiseTimeFrameHybrid(const int iteration) { LOGP(error, "initialiseTimeFrameHybrid: this method should never be called with CPU traits"); }
  virtual void computeTrackletsHybrid(const int iteration, int, int) { LOGP(error, "computeTrackletsHybrid: this method should never be called with CPU traits"); }
  virtual void computeCellsHybrid(const int iteration) { LOGP(error, "computeCellsHybrid: this method should never be called with CPU traits"); }
  virtual void findCellsNeighboursHybrid(const int iteration) { LOGP(error, "findCellsNeighboursHybrid: this method should never be called with CPU traits"); }
  virtual void findRoadsHybrid(const int iteration) { LOGP(error, "findRoadsHybrid: this method should never be called with CPU traits"); }
  virtual void findTracksHybrid(const int iteration) { LOGP(error, "findTracksHybrid: this method should never be called with CPU traits"); }
  virtual void findTracks() { LOGP(error, "findTracks: this method is deprecated."); }
  virtual void extendTracks(const int iteration);
  virtual void findShortPrimaries();
  virtual void setBz(float bz);
  virtual bool trackFollowing(TrackITSExt* track, int rof, bool outward, const int iteration);
  virtual void processNeighbours(int iLayer, int iLevel, const std::vector<CellSeed>& currentCellSeed, const std::vector<int>& currentCellId, std::vector<CellSeed>& updatedCellSeed, std::vector<int>& updatedCellId);

  void UpdateTrackingParameters(const std::vector<TrackingParameters>& trkPars);
  TimeFrame* getTimeFrame() { return mTimeFrame; }

  void setIsGPU(const unsigned char isgpu) { mIsGPU = isgpu; };
  float getBz() const;
  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType type) { mCorrType = type; }
  bool isMatLUT() const;

  // Others
  GPUhd() static consteval int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }
  template <bool print = false>
  const int4 getBinsRect(const Cluster&, int layer, float z1, float z2, float maxdeltaz, float maxdeltaphi) const noexcept;
  template <bool print = false>
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z, float maxdeltaz) const noexcept;
  template <bool print = false>
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z1, float z2, float maxdeltaz) const noexcept;
  void SetRecoChain(o2::gpu::GPUChainITS* chain) { mChain = chain; }
  void setSmoothing(bool v) { mApplySmoothing = v; }
  bool getSmoothing() const { return mApplySmoothing; }
  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }

  o2::gpu::GPUChainITS* getChain() const { return mChain; }

  // TimeFrame information forwarding
  virtual int getTFNumberOfClusters() const;
  virtual int getTFNumberOfTracklets() const;
  virtual int getTFNumberOfCells() const;

  float mBz = 5.f;

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const TrackingFrameInfo& tf3);
  bool fitTrack(TrackITSExt& track, int start, int end, int step, float chi2clcut = o2::constants::math::VeryBig, float chi2ndfcut = o2::constants::math::VeryBig, float maxQoverPt = o2::constants::math::VeryBig, int nCl = 0);

  int mNThreads = 1;
  bool mApplySmoothing = false;

 protected:
  o2::base::PropagatorImpl<float>::MatCorrType mCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
  o2::gpu::GPUChainITS* mChain = nullptr;
  TimeFrame* mTimeFrame;
  std::vector<TrackingParameters> mTrkParams;
  bool mIsGPU = false;
};

inline void TrackerTraits::initialiseTimeFrame(const int iteration)
{
  mTimeFrame->initialise(iteration, mTrkParams[iteration], mTrkParams[iteration].NLayers);
  setIsGPU(false);
}

inline float TrackerTraits::getBz() const
{
  return mBz;
}

inline void TrackerTraits::UpdateTrackingParameters(const std::vector<TrackingParameters>& trkPars)
{
  mTrkParams = trkPars;
}

template <bool print>
inline const int4 TrackerTraits::getBinsRect(const int layerIndex, float phi, float maxdeltaphi, float z, float maxdeltaz) const noexcept
{
  return getBinsRect<print>(layerIndex, phi, maxdeltaphi, z, z, maxdeltaz);
}

template <bool print>
inline const int4 TrackerTraits::getBinsRect(const Cluster& currentCluster, int layerIndex, float z1, float z2, float maxdeltaz, float maxdeltaphi) const noexcept
{
  return getBinsRect<print>(layerIndex, currentCluster.phi, maxdeltaphi, z1, z2, maxdeltaz);
}

template <bool print>
inline const int4 TrackerTraits::getBinsRect(const int layerIndex, float phi, float maxdeltaphi,
                                             float z1, float z2, float maxdeltaz) const noexcept
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = (maxdeltaphi > constants::math::Pi) ? 0.f : phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = (maxdeltaphi > constants::math::Pi) ? constants::math::TwoPi : phi + maxdeltaphi;

  if constexpr (print) {
    LOGP(info, "Requesting for layer {}; phi={} deltaphi={}; z1={} z2={} maxdeltaz={}", layerIndex, phi, maxdeltaphi, z1, z2, maxdeltaz);
    LOGP(info, "Ranges are z={}/{} phi={}/{}", zRangeMin, zRangeMax, phiRangeMin, phiRangeMax);
  }

  if (zRangeMax < -mTrkParams[0].LayerZ[layerIndex]) {
    if constexpr (print) {
      LOGP(info, "zRange failed zRangeMax(={}) < -LayerZ[idx=({})](={})", zRangeMax, layerIndex, -mTrkParams[0].LayerZ[layerIndex]);
    }
    return getEmptyBinsRect();
  }
  if (zRangeMin > mTrkParams[0].LayerZ[layerIndex]) {
    if constexpr (print) {
      LOGP(info, "zRange failed zRangeMin(={}) > LayerZ[idx=({})](={})", zRangeMin, layerIndex, mTrkParams[0].LayerZ[layerIndex]);
    }
    return getEmptyBinsRect();
  }
  if (zRangeMin > zRangeMax) {
    if constexpr (print) {
      LOGP(info, "zRange failed zRangeMin(={}) > zRangeMax(={})", zRangeMin, zRangeMax);
    }
    return getEmptyBinsRect();
  }

  const IndexTableUtils& utils{mTimeFrame->mIndexTableUtils};
  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(mTrkParams[0].ZBins - 1, utils.getZBinIndex(layerIndex, zRangeMax)), // /!\ trkParams can potentially change across iterations
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
