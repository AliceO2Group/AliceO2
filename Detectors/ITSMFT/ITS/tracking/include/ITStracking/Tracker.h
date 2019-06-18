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
#include "ITStracking/ROframe.h"
#include "ITStracking/MathUtils.h"
#include "ITStracking/PrimaryVertexContext.h"
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

class PrimaryVertexContext;
class TrackerTraits;

class Tracker
{

 public:
  Tracker(TrackerTraits* traits);

  Tracker(const Tracker&) = delete;
  Tracker& operator=(const Tracker&) = delete;
  ~Tracker();

  void setBz(float bz);
  float getBz() const;

  std::vector<TrackITSExt>& getTracks();
  dataformats::MCTruthContainer<MCCompLabel>& getTrackLabels();

  void clustersToTracks(const ROframe&, std::ostream& = std::cout);

  void setROFrame(std::uint32_t f) { mROFrame = f; }
  std::uint32_t getROFrame() const { return mROFrame; }
  void setParameters(const std::vector<MemoryParameters>&, const std::vector<TrackingParameters>&);

 private:
  track::TrackParCov buildTrackSeed(const Cluster& cluster1, const Cluster& cluster2, const Cluster& cluster3,
                                    const TrackingFrameInfo& tf3);
  template <typename... T>
  void initialisePrimaryVertexContext(T&&... args);
  void computeTracklets();
  void computeCells();
  void findCellsNeighbours(int& iteration);
  void findRoads(int& iteration);
  void findTracks(const ROframe& ev);
  bool fitTrack(const ROframe& event, TrackITSExt& track, int start, int end, int step);
  void traverseCellsTree(const int, const int);
  void computeRoadsMClabels(const ROframe&);
  void computeTracksMClabels(const ROframe&);

  template <typename... T>
  float evaluateTask(void (Tracker::*)(T...), const char*, std::ostream& ostream, T&&... args);

  TrackerTraits* mTraits = nullptr;                      /// Observer pointer, not owned by this class
  PrimaryVertexContext* mPrimaryVertexContext = nullptr; /// Observer pointer, not owned by this class

  std::vector<MemoryParameters> mMemParams;
  std::vector<TrackingParameters> mTrkParams;

  bool mCUDA = false;
  float mBz = 5.f;
  std::uint32_t mROFrame = 0;
  std::vector<TrackITSExt> mTracks;
  dataformats::MCTruthContainer<MCCompLabel> mTrackLabels;
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

inline void Tracker::setBz(float bz)
{
  mBz = bz;
}

template <typename... T>
void Tracker::initialisePrimaryVertexContext(T&&... args)
{
  mPrimaryVertexContext->initialise(std::forward<T>(args)...);
}

inline std::vector<TrackITSExt>& Tracker::getTracks()
{
  return mTracks;
}

inline dataformats::MCTruthContainer<MCCompLabel>& Tracker::getTrackLabels()
{
  return mTrackLabels;
}

template <typename... T>
float Tracker::evaluateTask(void (Tracker::*task)(T...), const char* taskName, std::ostream& ostream,
                            T&&... args)
{
  float diff{ 0.f };

  if (constants::DoTimeBenchmarks) {
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

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_TRACKER_H_ */
