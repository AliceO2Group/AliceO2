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
/// \file Vertexer.h
/// \brief
/// \author matteo.concas@cern.ch

#ifndef O2_ITS_TRACKING_VERTEXER_H_
#define O2_ITS_TRACKING_VERTEXER_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <array>
#include <iosfwd>
#include <memory>
#include <vector>

#include <oneapi/tbb/task_arena.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"

namespace o2::its
{

template <int NLayers>
class Vertexer
{
  using TimeFrameN = TimeFrame<NLayers>;
  using VertexerTraitsN = VertexerTraits<NLayers>;
  using LogFunc = std::function<void(const std::string& s)>;

 public:
  Vertexer(VertexerTraitsN* traits);
  virtual ~Vertexer() = default;
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

  void adoptTimeFrame(TimeFrameN& tf);
  auto& getVertParameters() const { return mTraits->getVertexingParameters(); }
  void setParameters(const std::vector<VertexingParameters>& vertParams) { mVertParams = vertParams; }
  const auto& getParameters() const noexcept { return mVertParams; }
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource> pool) { mMemoryPool = pool; }

  float clustersToVertices(LogFunc = [](const std::string& s) { std::cout << s << '\n'; });
  void filterMCTracklets();
  void printSummary() const;

  template <typename... T>
  void findTracklets(T&&... args)
  {
    mTraits->computeTracklets(std::forward<T>(args)...);
  }
  template <typename... T>
  void validateTracklets(T&&... args)
  {
    mTraits->computeTrackletMatching(std::forward<T>(args)...);
  }
  template <typename... T>
  void findVertices(T&&... args)
  {
    mTraits->computeVertices(std::forward<T>(args)...);
  }

  void addTruthSeeds() { mTraits->addTruthSeedingVertices(); }

  template <typename... T>
  void initialiseVertexer(T&&... args)
  {
    mTraits->initialise(std::forward<T>(args)...);
  }
  template <typename... T>
  void initialiseTimeFrame(T&&... args);

  void sortVertices();

  // Utils
  template <typename... T>
  float evaluateTask(void (Vertexer::*task)(T...), std::string_view taskName, int iteration, LogFunc& logger, T&&... args);

  void printEpilog(LogFunc& logger,
                   unsigned int trackletN01, unsigned int trackletN12,
                   unsigned selectedN, unsigned int vertexN, unsigned int totalVertexN,
                   float initT, float trackletT, float selecT, float vertexT);

  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena) { mTraits->setNThreads(n, arena); }

 private:
  std::uint32_t mTimeFrameCounter = 0;

  VertexerTraitsN* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrameN* mTimeFrame = nullptr;   /// Observer pointer, not owned by this class

  std::vector<VertexingParameters> mVertParams;
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;

  enum Steps {
    Init = 0,
    Trackleting,
    Selection,
    Finding,
    TruthSeeding,
    NSteps,
  };
  Steps mCurStep{Init};
  static constexpr std::array<const char*, NSteps> StateNames{"Initialisation", "Tracklet finding", "Tracklet selection", "Vertex finding", "Truth seeding"};
  std::vector<std::array<TimingStats, NSteps>> mTimingStats;
  void addTimingStatCurStep(int iteration, double timeMs);
};

template <int NLayers>
template <typename... T>
float Vertexer<NLayers>::evaluateTask(void (Vertexer<NLayers>::*task)(T...), std::string_view taskName, int iteration, LogFunc& logger, T&&... args)
{
  float diff{0.f};

  if (mVertParams[iteration].PrintMemory) {
    mMemoryPool->resetPeakMemory();
  }

  if constexpr (constants::DoTimeBenchmarks) {
    auto start = std::chrono::high_resolution_clock::now();
    (this->*task)(std::forward<T>(args)...);
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

    if (mVertParams[iteration].SaveTimeBenchmarks) {
      std::string taskNameStr(taskName);
      std::transform(taskNameStr.begin(), taskNameStr.end(), taskNameStr.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::replace(taskNameStr.begin(), taskNameStr.end(), ' ', '_');
      if (std::ofstream file{"its_time_benchmarks.txt", std::ios::app}) {
        file << "vtx:" << iteration << '\t' << taskNameStr << '\t' << diff << '\n';
      }
      addTimingStatCurStep(iteration, diff);
    }
  } else {
    (this->*task)(std::forward<T>(args)...);
  }

  if (mVertParams[iteration].PrintMemory) {
    LOGP(info, "iter:{}:{}: {}", iteration, StateNames[mCurStep], mMemoryPool->asString());
  }

  return diff;
}

} // namespace o2::its
#endif
