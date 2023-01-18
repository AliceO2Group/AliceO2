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
typedef std::function<int(o2::gpu::GPUChainITS&, std::vector<Road>& roads, std::vector<const Cluster*>&, std::vector<const Cell*>&, const std::vector<std::vector<TrackingFrameInfo>>&, std::vector<TrackITSExt>&)> FuncRunITSTrackFit_t;

class TrackerTraits
{
 public:
  virtual ~TrackerTraits() = default;
  virtual void adoptTimeFrame(TimeFrame* tf);
  virtual void initialiseTimeFrame(const int iteration);
  virtual void computeLayerTracklets(const int iteration);
  virtual void computeLayerCells(const int iteration);
  virtual void findCellsNeighbours(const int iteration);
  virtual void findRoads(const int iteration);
  virtual void findTracks();
  virtual void extendTracks(const int iteration);
  virtual void findShortPrimaries();
  virtual void refitTracks(const int iteration, const std::vector<std::vector<TrackingFrameInfo>>&, std::vector<TrackITSExt>&);
  virtual void setBz(float bz);
  virtual bool trackFollowing(TrackITSExt* track, int rof, bool outward, const int iteration);

  void UpdateTrackingParameters(const std::vector<TrackingParameters>& trkPars);
  TimeFrame* getTimeFrame() { return mTimeFrame; }

  void setIsGPU(const unsigned char isgpu) { mIsGPU = isgpu; };
  float getBz() const;
  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType type) { mCorrType = type; }
  bool isMatLUT() const;

  // Others
  GPUhd() static constexpr int4 getEmptyBinsRect() { return int4{0, 0, 0, 0}; }
  const int4 getBinsRect(const Cluster&, int layer, float z1, float z2, float maxdeltaz, float maxdeltaphi);
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z, float maxdeltaz);
  const int4 getBinsRect(int layer, float phi, float maxdeltaphi, float z1, float z2, float maxdeltaz);
  void SetRecoChain(o2::gpu::GPUChainITS* chain, FuncRunITSTrackFit_t&& funcRunITSTrackFit)
  {
    mChainRunITSTrackFit = funcRunITSTrackFit;
    mChain = chain;
  }
  void setSmoothing(bool v) { mApplySmoothing = v; }
  bool getSmoothing() const { return mApplySmoothing; }
  void setNThreads(int n);
  int getNThreads() const { return mNThreads; }

  // TimeFrame information forwarding
  virtual int getTFNumberOfClusters() const;
  virtual int getTFNumberOfTracklets() const;
  virtual int getTFNumberOfCells() const;

  float mBz = 5.f;

 private:
  void traverseCellsTree(const int, const int);
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3, const TrackingFrameInfo& tf3, float misalignmentY, float misalignmentZ);
  bool fitTrack(TrackITSExt& track, int start, int end, int step, const float chi2cut = o2::constants::math::VeryBig, const float maxQoverPt = o2::constants::math::VeryBig);

  int mNThreads = 1;
  bool mApplySmoothing = false;
  o2::base::PropagatorImpl<float>::MatCorrType mCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;

  // virtual bool checkTFMemory(const int iteration);

 protected:
  TimeFrame* mTimeFrame;
  std::vector<TrackingParameters> mTrkParams;
  bool mIsGPU = false;
  o2::gpu::GPUChainITS* mChain = nullptr;
  FuncRunITSTrackFit_t mChainRunITSTrackFit;
};

inline float TrackerTraits::getBz() const
{
  return mBz;
}

inline void TrackerTraits::UpdateTrackingParameters(const std::vector<TrackingParameters>& trkPars)
{
  mTrkParams = trkPars;
}

inline const int4 TrackerTraits::getBinsRect(const int layerIndex, float phi, float maxdeltaphi, float z, float maxdeltaz)
{
  return getBinsRect(layerIndex, phi, maxdeltaphi, z, z, maxdeltaz);
}

inline const int4 TrackerTraits::getBinsRect(const Cluster& currentCluster, int layerIndex, float z1, float z2, float maxdeltaz, float maxdeltaphi)
{
  return getBinsRect(layerIndex, currentCluster.phi, maxdeltaphi, z1, z2, maxdeltaz);
}

inline void TrackerTraits::initialiseTimeFrame(const int iteration)
{
  mTimeFrame->initialise(iteration, mTrkParams[iteration], 7);
  setIsGPU(false);
}

inline const int4 TrackerTraits::getBinsRect(const int layerIndex, float phi, float maxdeltaphi,
                                             float z1, float z2, float maxdeltaz)
{
  const float zRangeMin = o2::gpu::GPUCommonMath::Min(z1, z2) - maxdeltaz;
  const float phiRangeMin = phi - maxdeltaphi;
  const float zRangeMax = o2::gpu::GPUCommonMath::Max(z1, z2) + maxdeltaz;
  const float phiRangeMax = phi + maxdeltaphi;

  if (zRangeMax < -mTrkParams[0].LayerZ[layerIndex + 1] ||
      zRangeMin > mTrkParams[0].LayerZ[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  const IndexTableUtils& utils{mTimeFrame->mIndexTableUtils};
  return int4{o2::gpu::GPUCommonMath::Max(0, utils.getZBinIndex(layerIndex + 1, zRangeMin)),
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMin)),
              o2::gpu::GPUCommonMath::Min(mTrkParams[0].ZBins - 1, utils.getZBinIndex(layerIndex + 1, zRangeMax)), // /!\ trkParams can potentially change across iterations
              utils.getPhiBinIndex(math_utils::getNormalizedPhi(phiRangeMax))};
}
} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKERTRAITS_H_ */
