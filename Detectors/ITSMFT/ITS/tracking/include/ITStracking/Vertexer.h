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

#include <oneapi/tbb/task_arena.h>

#include "ITStracking/Constants.h"
#include "ITStracking/Definitions.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/VertexerTraits.h"
#include "ITStracking/BoundedAllocator.h"

namespace o2::its
{

template <int nLayers>
class Vertexer
{
  using TimeFrameN = TimeFrame<nLayers>;
  using VertexerTraitsN = VertexerTraits<nLayers>;
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
  void setMemoryPool(std::shared_ptr<BoundedMemoryResource>& pool) { mMemoryPool = pool; }

  std::vector<Vertex> exportVertices();
  VertexerTraitsN* getTraits() const { return mTraits; };

  float clustersToVertices(LogFunc = [](const std::string& s) { std::cout << s << '\n'; });
  void filterMCTracklets();

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

  // Utils
  template <typename... T>
  float evaluateTask(void (Vertexer::*task)(T...), std::string_view taskName, int iteration, LogFunc& logger, T&&... args);

  void printEpilog(LogFunc& logger,
                   const unsigned int trackletN01, const unsigned int trackletN12,
                   const unsigned selectedN, const unsigned int vertexN, const float initT,
                   const float trackletT, const float selecT, const float vertexT);

  void setNThreads(int n, std::shared_ptr<tbb::task_arena>& arena) { mTraits->setNThreads(n, arena); }

 private:
  std::uint32_t mTimeFrameCounter = 0;

  VertexerTraitsN* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrameN* mTimeFrame = nullptr;   /// Observer pointer, not owned by this class

  std::vector<VertexingParameters> mVertParams;
  std::shared_ptr<BoundedMemoryResource> mMemoryPool;

  enum State {
    Init = 0,
    Trackleting,
    Validating,
    Finding,
    TruthSeeding,
    NStates,
  };
  State mCurState{Init};
  static constexpr std::array<const char*, NStates> StateNames{"Initialisation", "Tracklet finding", "Tracklet validation", "Vertex finding", "Truth seeding"};
};

template <int nLayers>
template <typename... T>
float Vertexer<nLayers>::evaluateTask(void (Vertexer<nLayers>::*task)(T...), std::string_view taskName, int iteration, LogFunc& logger, T&&... args)
{
  float diff{0.f};

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

    if (mVertParams[0].SaveTimeBenchmarks) {
      std::string taskNameStr(taskName);
      std::transform(taskNameStr.begin(), taskNameStr.end(), taskNameStr.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::replace(taskNameStr.begin(), taskNameStr.end(), ' ', '_');
      if (std::ofstream file{"its_time_benchmarks.txt", std::ios::app}) {
        file << "vtx:" << iteration << '\t' << taskNameStr << '\t' << diff << '\n';
      }
    }
  } else {
    (this->*task)(std::forward<T>(args)...);
  }

  return diff;
}

} // namespace o2::its
#endif
