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
#include "DetectorsBase/MatLayerCylSet.h"
#include "CommonConstants/MathConstants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/ROframe.h"
#include "ITStracking/MathUtils.h"
#include "DetectorsBase/Propagator.h"
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
  void setBz(float bz);
  float getBz() const;

  void clustersToTracks(
    std::function<void(std::string s)> = [](std::string s) { std::cout << s << std::endl; }, std::function<void(std::string s)> = [](std::string s) { std::cerr << s << std::endl; });
  void clustersToTracksGPU(std::function<void(std::string s)> = [](std::string s) { std::cout << s << std::endl; });
  void setSmoothing(bool v) { mApplySmoothing = v; }
  bool getSmoothing() const { return mApplySmoothing; }

  std::vector<TrackITSExt>& getTracks();

  void setCorrType(const o2::base::PropagatorImpl<float>::MatCorrType& type) { mCorrType = type; }
  void setParameters(const std::vector<MemoryParameters>&, const std::vector<TrackingParameters>&);
  void getGlobalConfiguration();
  bool isMatLUT() const { return o2::base::Propagator::Instance()->getMatLUT() && (mCorrType == o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrLUT); }
  // GPU-specific interfaces
  TimeFrame* getTimeFrameGPU();
  void loadToDevice();

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3,
                                    const TrackingFrameInfo& tf3, float resolution);
  template <typename... T>
  void initialiseTimeFrame(T&&... args);
  void computeTracklets();
  void computeCells();
  void findCellsNeighbours(int& iteration);
  void findRoads(int& iteration);
  void findTracks();
  void extendTracks();
  bool fitTrack(TrackITSExt& track, int start, int end, int step, const float chi2cut = o2::constants::math::VeryBig, const float maxQoverPt = o2::constants::math::VeryBig);
  void traverseCellsTree(const int, const int);
  void computeRoadsMClabels();
  void computeTracksMClabels();
  void rectifyClusterIndices();

  template <typename... T>
  float evaluateTask(void (Tracker::*)(T...), const char*, std::function<void(std::string s)> logger, T&&... args);

  TrackerTraits* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrame* mTimeFrame = nullptr;  /// Observer pointer, not owned by this class

  std::vector<MemoryParameters> mMemParams;
  std::vector<TrackingParameters> mTrkParams;

  bool mCUDA = false;
  bool mApplySmoothing = false;
  o2::base::PropagatorImpl<float>::MatCorrType mCorrType = o2::base::PropagatorImpl<float>::MatCorrType::USEMatCorrNONE;
  float mBz = 5.f;
  std::uint32_t mTimeFrameCounter = 0;
  o2::gpu::GPUChainITS* mRecoChain = nullptr;

};

inline void Tracker::setParameters(const std::vector<MemoryParameters>& memPars, const std::vector<TrackingParameters>& trkPars)
{
  mMemParams = memPars;
  mTrkParams = trkPars;
}

inline float Tracker::getBz() const
{
  return mBz;
}

template <typename... T>
void Tracker::initialiseTimeFrame(T&&... args)
{
  mTimeFrame->initialise(std::forward<T>(args)...);
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
