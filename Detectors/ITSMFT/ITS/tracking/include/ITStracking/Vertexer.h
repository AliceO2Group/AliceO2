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
#include "ITStracking/VertexerTraits.h"
#include "ReconstructionDataFormats/Vertex.h"

#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"

class TTree;

namespace o2
{
namespace its
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class Vertexer
{
 public:
  Vertexer(VertexerTraits* traits);
  virtual ~Vertexer() = default;
  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

  void setROframe(const uint32_t ROframe) { mROframe = ROframe; }
  void setParameters(const VertexingParameters& verPar);
  VertexingParameters getVertParameters() const;

  uint32_t getROFrame() const { return mROframe; }
  std::vector<Vertex> exportVertices();
  VertexerTraits* getTraits() const { return mTraits; };

  float clustersToVertices(ROframe&, const bool useMc = false, std::ostream& = std::cout);
  void filterMCTracklets();
  void validateTracklets();

  template <typename... T>
  void findTracklets(T&&... args);

  void findTrivialMCTracklets();
  void findVertices();

  template <typename... T>
  void initialiseVertexer(T&&... args);

  // Utils
  void dumpTraits();
  template <typename... T>
  float evaluateTask(void (Vertexer::*)(T...), const char*, std::ostream& ostream, T&&... args);

  // debug
  void setDebugCombinatorics();
  void setDebugTrackletSelection();
  void setDebugLines();
  void setDebugSummaryLines();
  // \debug

 private:
  std::uint32_t mROframe = 0;
  VertexerTraits* mTraits = nullptr;
};

#ifdef _ALLOW_DEBUG_TREES_ITS_
inline void Vertexer::filterMCTracklets()
{
  mTraits->computeMCFiltering();
}
#endif

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

inline void Vertexer::findTrivialMCTracklets()
{
  mTraits->computeTrackletsPureMontecarlo();
}

inline VertexingParameters Vertexer::getVertParameters() const
{
  return mTraits->getVertexingParameters();
}

inline void Vertexer::setParameters(const VertexingParameters& verPar)
{
  mTraits->updateVertexingParameters(verPar);
}

inline void Vertexer::dumpTraits()
{
  mTraits->dumpVertexerTraits();
}

inline void Vertexer::validateTracklets()
{
  mTraits->computeTrackletMatching();
}

inline std::vector<Vertex> Vertexer::exportVertices()
{
  std::vector<Vertex> vertices;
  for (auto& vertex : mTraits->getVertices()) {
    std::cout << "\t\tFound vertex with: " << std::setw(6) << vertex.mContributors << " contributors" << std::endl;
    vertices.emplace_back(Point3D<float>(vertex.mX, vertex.mY, vertex.mZ), vertex.mRMS2, vertex.mContributors, vertex.mAvgDistance2);
    vertices.back().setTimeStamp(vertex.mTimeStamp);
  }
  return vertices;
}

template <typename... T>
float Vertexer::evaluateTask(void (Vertexer::*task)(T...), const char* taskName, std::ostream& ostream,
                             T&&... args)
{
  float diff{0.f};

  if (constants::DoTimeBenchmarks) {
    auto start = std::chrono::high_resolution_clock::now();
    (this->*task)(std::forward<T>(args)...);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> diff_t{end - start};
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

inline void Vertexer::setDebugCombinatorics()
{
  mTraits->setDebugFlag(VertexerDebug::CombinatoricsTreeAll);
}

inline void Vertexer::setDebugTrackletSelection()
{
  mTraits->setDebugFlag(VertexerDebug::TrackletTreeAll);
}

inline void Vertexer::setDebugLines()
{
  mTraits->setDebugFlag(VertexerDebug::LineTreeAll);
}

inline void Vertexer::setDebugSummaryLines()
{
  mTraits->setDebugFlag(VertexerDebug::LineSummaryAll);
}

} // namespace its
} // namespace o2
#endif
