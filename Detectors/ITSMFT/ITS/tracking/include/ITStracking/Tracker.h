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
/// \file Tracker.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_TRACKER_H_
#define TRACKINGITSU_INCLUDE_TRACKER_H_

#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string_view>
#include <utility>
#include <sstream>

#include <oneapi/tbb/task_arena.h>

#include "ITStracking/Configuration.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/TrackerTraits.h"
#include "ITStracking/BoundedAllocator.h"

namespace o2
{

namespace gpu
{
class GPUChainITS;
}
namespace its
{

template <int NLayers>
class Tracker
{
  using LogFunc = std::function<void(const std::string& s)>;

 public:
  Tracker(TrackerTraits<NLayers>* traits);

  void adoptTimeFrame(TimeFrame<NLayers>& tf);

  float clustersToTracks(
    const LogFunc& = [](const std::string& s) { std::cout << s << '\n'; },
    const LogFunc& = [](const std::string& s) { std::cerr << s << '\n'; });

  void setParameters(const std::vector<TrackingParameters>& p) { mTrkParams = p; }
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool) { mMemoryPool = pool; }
  std::vector<TrackingParameters>& getParameters() { return mTrkParams; }
  void setBz(float bz) { mTraits->setBz(bz); }
  void setTimeSlice(size_t slice) noexcept { mTimeSlice = slice; }
  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena) { mTraits->setNThreads(n, arena); }
  void printSummary() const;
  void computeTracksMClabels();

 private:
  void initialiseTimeFrame(int iteration) { mTraits->initialiseTimeFrame(iteration); }
  void computeTracklets(int iteration, int iVertex) { mTraits->computeLayerTracklets(iteration, iVertex); }
  void computeCells(int iteration) { mTraits->computeLayerCells(iteration); }
  void findCellsNeighbours(int iteration) { mTraits->findCellsNeighbours(iteration); }
  void findRoads(int iteration) { mTraits->findRoads(iteration); }

  void rectifyClusterIndices();
  void sortTracks();

  template <typename... T, typename... F>
  float evaluateTask(void (Tracker::*task)(T...), std::string_view taskName, int iteration, const LogFunc& logger, F&&... args);

  TrackerTraits<NLayers>* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrame<NLayers>* mTimeFrame = nullptr;  /// Observer pointer, not owned by this class

  std::vector<TrackingParameters> mTrkParams;
  o2::gpu::GPUChainITS* mRecoChain = nullptr;

  size_t mTimeSlice{0}; // current timeslice
  unsigned int mNumberOfDroppedTFs{0};
  unsigned int mTimeFrameCounter{0};
  double mTotalTime{0};
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;

  enum State {
    TFInit = 0,
    Trackleting,
    Celling,
    Neighbouring,
    Roading,
    NStates,
  };
  State mCurState{TFInit};
  static constexpr std::array<const char*, NStates> StateNames{"TimeFrame initialisation", "Tracklet finding", "Cell finding", "Neighbour finding", "Road finding"};
};

template <int NLayers>
template <typename... T, typename... F>
float Tracker<NLayers>::evaluateTask(void (Tracker<NLayers>::*task)(T...), std::string_view taskName, int iteration, const LogFunc& logger, F&&... args)
{
  float diff{0.f};

  if constexpr (constants::DoTimeBenchmarks) {
    auto start = std::chrono::high_resolution_clock::now();
    (this->*task)(std::forward<F>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_t{end - start};
    diff = diff_t.count();

    std::stringstream sstream;
    if (taskName.empty()) {
      sstream << diff << "\t";
    } else {
      sstream << std::setw(2) << " - " << taskName << " completed in: " << diff << " ms";
    }
    logger(sstream.str());

    if (mTrkParams[0].SaveTimeBenchmarks) {
      std::string taskNameStr(taskName);
      std::transform(taskNameStr.begin(), taskNameStr.end(), taskNameStr.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::replace(taskNameStr.begin(), taskNameStr.end(), ' ', '_');
      if (std::ofstream file{"its_time_benchmarks.txt", std::ios::app}) {
        file << "trk:" << iteration << '\t' << taskNameStr << '\t' << diff << '\n';
      }
    }

  } else {
    (this->*task)(std::forward<F>(args)...);
  }

  if (mTrkParams[iteration].PrintMemory) {
    LOGP(info, "iter:{}:{}: {}", iteration, StateNames[mCurState], mMemoryPool->asString());
  }

  return diff;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
