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
#include <iostream>
#include <memory>
#include <utility>

#include "ITSReconstruction/CA/Definitions.h"
#include "ITSReconstruction/CA/Event.h"
#include "ITSReconstruction/CA/MathUtils.h"
#include "ITSReconstruction/CA/PrimaryVertexContext.h"
#include "ITSReconstruction/CA/Road.h"

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
  void computeLayerTracklets(PrimaryVertexContext&);
  void computeLayerCells(PrimaryVertexContext&);

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

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3,
                                    const TrackingFrameInfo& tf3);
  template <typename... T>
  void initialisePrimaryVertexContext(T&&... args);
  void computeTracklets();
  void computeCells();
  void findCellsNeighbours();
  void findRoads();
  void findTracks(const Event& ev);
  bool fitTrack(const Event& event, TrackITS& track, int start, int end, int step);
  void traverseCellsTree(const int, const int);
  void computeRoadsMClabels(const Event&);
  void computeTracksMClabels(const Event&);

  template <typename... T>
  float evaluateTask(void (Tracker<IsGPU>::*)(T...), const char*, std::ostream& ostream, T&&... args);

  float mBz = 5.f;
  PrimaryVertexContext mPrimaryVertexContext;
  std::vector<TrackITS> mTracks;
  dataformats::MCTruthContainer<MCCompLabel> mTrackLabels;
};

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
  mPrimaryVertexContext.initialise(std::forward<T>(args)...);
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
void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerTracklets(PrimaryVertexContext&);
template <>
void TrackerTraits<TRACKINGITSU_GPU_MODE>::computeLayerCells(PrimaryVertexContext&);
}
}
}

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
