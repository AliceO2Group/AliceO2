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
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <utility>
#include <sstream>

#include "ITStracking/Configuration.h"
#include "CommonConstants/MathConstants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/Road.h"

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{

namespace gpu
{
class GPUChainITS;
}
namespace its
{
class TrackerTraits;

class Tracker
{

 public:
  Tracker(TrackerTraits* traits);

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;
  ~Tracker();

  void adoptTimeFrame(TimeFrame& tf);

  void clustersToTracks(
    std::function<void(std::string s)> = [](std::string s) { std::cout << s << std::endl; }, std::function<void(std::string s)> = [](std::string s) { std::cerr << s << std::endl; });
  std::vector<TrackITSExt>& getTracks();

  void setParameters(const std::vector<TrackingParameters>&);
  std::vector<TrackingParameters>& getParameters() { return mTrkParams; }
  void getGlobalConfiguration();
  void setBz(float);
  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType type);
  bool isMatLUT() const;
  void setNThreads(int n);
  int getNThreads() const;
  std::uint32_t mTimeFrameCounter = 0;

 private:
  void initialiseTimeFrame(int& iteration);
  void computeTracklets(int& iteration);
  void computeCells(int& iteration);
  void findCellsNeighbours(int& iteration);
  void findRoads(int& iteration);
  void findShortPrimaries();
  void findTracks();
  void extendTracks(int& iteration);

  // MC interaction
  void computeRoadsMClabels();
  void computeTracksMClabels();
  void rectifyClusterIndices();

  template <typename... T>
  float evaluateTask(void (Tracker::*)(T...), const char*, std::function<void(std::string s)> logger, T&&... args);

  TrackerTraits* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrame* mTimeFrame = nullptr;  /// Observer pointer, not owned by this class

  std::vector<TrackingParameters> mTrkParams;
  o2::gpu::GPUChainITS* mRecoChain = nullptr;

  unsigned int mNumberOfRuns{0};
};

inline void Tracker::setParameters(const std::vector<TrackingParameters>& trkPars)
{
  mTrkParams = trkPars;
}

template <typename... T>
float Tracker::evaluateTask(void (Tracker::*task)(T...), const char* taskName, std::function<void(std::string s)> logger,
                            T&&... args)
{
  float diff{0.f};

  if (constants::DoTimeBenchmarks) {
    auto start = std::chrono::high_resolution_clock::now();
    (this->*task)(std::forward<T>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_t{end - start};
    diff = diff_t.count();

    std::stringstream sstream;
    if (taskName == nullptr) {
      sstream << diff << "\t";
    } else {
      sstream << std::setw(2) << " - " << taskName << " completed in: " << diff << " ms";
    }
    logger(sstream.str());
  } else {
    (this->*task)(std::forward<T>(args)...);
  }

  return diff;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
