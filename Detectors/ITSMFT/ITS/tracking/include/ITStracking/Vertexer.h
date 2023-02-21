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

#include "ITStracking/ROframe.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/TimeFrame.h"
#include "ITStracking/VertexerTraits.h"
#include "ReconstructionDataFormats/Vertex.h"

#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"

#include "GPUCommonLogger.h"

class TTree;

namespace o2
{
namespace its
{
using TimeFrame = o2::its::TimeFrame;
using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class Vertexer
{
 public:
  Vertexer(VertexerTraits* traits);
  virtual ~Vertexer() = default;
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

  void adoptTimeFrame(TimeFrame& tf);
  void setParameters(const VertexingParameters& verPar);
  VertexingParameters& getVertParameters() const;
  void getGlobalConfiguration();

  std::vector<Vertex> exportVertices();
  VertexerTraits* getTraits() const { return mTraits; };

  float clustersToVertices(std::function<void(std::string s)> = [](std::string s) { std::cout << s << std::endl; });
  void filterMCTracklets();
  void validateTracklets();

  template <typename... T>
  void findTracklets(T&&... args);

  void findTrivialMCTracklets();
  void findVertices();
  void findHistVertices();

  template <typename... T>
  void initialiseVertexer(T&&... args);
  template <typename... T>
  void initialiseTimeFrame(T&&... args);

  // Utils
  void dumpTraits();
  template <typename... T>
  float evaluateTask(void (Vertexer::*)(T...), const char*, std::function<void(std::string s)> logger, T&&... args);
  void printEpilog(std::function<void(std::string s)> logger, const float total);

 private:
  std::uint32_t mTimeFrameCounter = 0;

  VertexerTraits* mTraits = nullptr; /// Observer pointer, not owned by this class
  TimeFrame* mTimeFrame = nullptr;   /// Observer pointer, not owned by this class
};

template <typename... T>
void Vertexer::initialiseVertexer(T&&... args)
{
  mTraits->initialise(std::forward<T>(args)...);
}

template <typename... T>
void Vertexer::findTracklets(T&&... args)
{
  mTraits->computeTracklets(std::forward<T>(args)...);
}

inline void Vertexer::setParameters(const VertexingParameters& verPar)
{
  mTraits->updateVertexingParameters(verPar);
}

inline VertexingParameters& Vertexer::getVertParameters() const
{
  return mTraits->getVertexingParameters();
}

inline void Vertexer::dumpTraits()
{
  mTraits->dumpVertexerTraits();
}

inline void Vertexer::validateTracklets()
{
  mTraits->computeTrackletMatching();
}

template <typename... T>
float Vertexer::evaluateTask(void (Vertexer::*task)(T...), const char* taskName, std::function<void(std::string s)> logger,
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
#endif
