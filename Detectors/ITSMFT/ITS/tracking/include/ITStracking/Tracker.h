// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Tracker.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKER_H_
#define TRACKINGITSU_INCLUDE_TRACKER_H_

#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>

#include "ITStracking/Configuration.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Event.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/PrimaryVertexContext.h"
#include "ITStracking/Road.h"

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace ITS
{
namespace CA
{

template <bool IsGPU>
class TrackerTraits
{
 public:
  GPU_HOST_DEVICE constexpr int4 getEmptyBinsRect() { return int4{ 0, 0, 0, 0 }; }
  GPU_DEVICE const int4 getBinsRect(const Cluster&, const int, const float, float maxdeltaz, float maxdeltaphi);

  void computeLayerTracklets(PrimaryVertexContext&, const TrackingParameters& trkPars, int iteration = 0);
  void computeLayerCells(PrimaryVertexContext&, const TrackingParameters& trkPars, int iteration = 0);

 protected:
  ~TrackerTraits() = default;
};

template <bool IsGPU>
class Tracker : private TrackerTraits<IsGPU>
{
 private:
  typedef TrackerTraits<IsGPU> Trait;

 public:
  Tracker();

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;

  void setBz(float bz);
  float getBz() const;

  std::vector<TrackITS>& getTracks();
  dataformats::MCTruthContainer<MCCompLabel>& getTrackLabels();

  void clustersToTracks(const Event&, std::ostream& = std::cout);

  void setROFrame(std::uint32_t f) { mROFrame = f; }
  std::uint32_t getROFrame() const { return mROFrame; }
  void setParameters(const MemoryParameters&, const TrackingParameters&);

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3,
                                    const TrackingFrameInfo& tf3);
  template <typename... T>
  void initialisePrimaryVertexContext(T&&... args);
  void computeTracklets(int& iteration);
  void computeCells(int& iteration);
  void findCellsNeighbours(int& iteration);
  void findRoads(int& iteration);
  void findTracks(const Event& ev);
  bool fitTrack(const Event& event, TrackITS& track, int start, int end, int step);
  void traverseCellsTree(const int, const int);
  void computeRoadsMClabels(const Event&);
  void computeTracksMClabels(const Event&);

  template <typename... T>
  float evaluateTask(void (Tracker<IsGPU>::*)(T...), const char*, std::ostream& ostream, T&&... args);

  MemoryParameters mMemParams;
  TrackingParameters mTrkParams;

  float mBz = 5.f;
  std::uint32_t mROFrame = 0;
  PrimaryVertexContext mPrimaryVertexContext;
  std::vector<TrackITS> mTracks;
  dataformats::MCTruthContainer<MCCompLabel> mTrackLabels;
};

template <bool IsGPU>
GPU_DEVICE const int4 TrackerTraits<IsGPU>::getBinsRect(const Cluster& currentCluster, const int layerIndex,
                                                        const float directionZIntersection, float maxdeltaz, float maxdeltaphi)
{
  const float zRangeMin = directionZIntersection - 2 * maxdeltaz;
  const float phiRangeMin = currentCluster.phiCoordinate - maxdeltaphi;
  const float zRangeMax = directionZIntersection + 2 * maxdeltaz;
  const float phiRangeMax = currentCluster.phiCoordinate + maxdeltaphi;

  if (zRangeMax < -Constants::ITS::LayersZCoordinate()[layerIndex + 1] ||
      zRangeMin > Constants::ITS::LayersZCoordinate()[layerIndex + 1] || zRangeMin > zRangeMax) {

    return getEmptyBinsRect();
  }

  return int4{ MATH_MAX(0, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMin)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMin)),
               MATH_MIN(Constants::IndexTable::ZBins - 1, IndexTableUtils::getZBinIndex(layerIndex + 1, zRangeMax)),
               IndexTableUtils::getPhiBinIndex(MathUtils::getNormalizedPhiCoordinate(phiRangeMax)) };
}

template <bool IsGPU>
void Tracker<IsGPU>::setParameters(const MemoryParameters& memPars, const TrackingParameters& trkPars)
{
  mMemParams = memPars;
  mTrkParams = trkPars;
}

template <bool IsGPU>
float Tracker<IsGPU>::getBz() const
{
  return mBz;
}

template <bool IsGPU>
void Tracker<IsGPU>::setBz(float bz)
{
  mBz = bz;
}

template <bool IsGPU>
template <typename... T>
void Tracker<IsGPU>::initialisePrimaryVertexContext(T&&... args)
{
  mPrimaryVertexContext.initialise(mMemParams, std::forward<T>(args)...);
}

template <bool IsGPU>
inline std::vector<TrackITS>& Tracker<IsGPU>::getTracks()
{
  return mTracks;
}

template <bool IsGPU>
inline dataformats::MCTruthContainer<MCCompLabel>& Tracker<IsGPU>::getTrackLabels()
{
  return mTrackLabels;
}

template <bool IsGPU>
template <typename... T>
float Tracker<IsGPU>::evaluateTask(void (Tracker<IsGPU>::*task)(T...), const char* taskName, std::ostream& ostream,
                                   T&&... args)
{
  float diff{ 0.f };

  if (Constants::DoTimeBenchmarks) {
    auto start = std::chrono::high_resolution_clock::now();
    (this->*task)(std::forward<T>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_t{ end - start };
    diff = diff_t.count();

    if (taskName == nullptr) {
      ostream << diff << "\t";
    } else {
      ostream << std::setw(2) << " - " << taskName << " completed in: " << diff << " ms" << std::endl;
    }
  } else {
    (this->*task)(std::forward<T>(args)...);
  }

  return diff;
}

template <>
void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerTracklets(PrimaryVertexContext&, const TrackingParameters& trkPars, int iteration);
template <>
void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerCells(PrimaryVertexContext&, const TrackingParameters& trkPars, int iteration);
} // namespace CA
} // namespace ITS
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
