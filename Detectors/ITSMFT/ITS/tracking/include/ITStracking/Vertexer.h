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
///

#ifndef O2_ITS_TRACKING_VERTEXER_H_
#define O2_ITS_TRACKING_VERTEXER_H_

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <array>
#include <iosfwd>

#include "ITStracking/ROframe.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/VertexerTraits.h"
#include "ReconstructionDataFormats/Vertex.h"

namespace o2
{
namespace ITS
{

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

class Vertexer
{
 public:
  Vertexer(VertexerTraits* traits);

  Vertexer(const Vertexer&) = delete;
  Vertexer& operator=(const Vertexer&) = delete;

  void setROframe(const uint32_t ROframe) { mROframe = ROframe; }
  void setParameters(const VertexingParameters& verPar) { mVertParams = verPar; }

  uint32_t getROFrame() const { return mROframe; }
  std::vector<Vertex> exportVertices();
  VertexingParameters getVertParameters() const { return mVertParams; }
  VertexerTraits* getTraits() const { return mTraits; };

  void clustersToVertices(ROframe&, std::ostream& = std::cout);
  template <typename... T>
  void initialiseVertexer(T&&... args);
  void findTracklets(const bool useMCLabel = false);
  void findVertices();
  // void writeEvent(ROframe*);

  // Utils
  void dumpTraits();
  template <typename... T>
  float evaluateTask(void (Vertexer::*)(T...), const char*, std::ostream& ostream, T&&... args);

 private:
  // ROframe* mFrame;
  std::uint32_t mROframe = 0;
  VertexerTraits* mTraits = nullptr;
  VertexingParameters mVertParams;
};

template <typename... T>
void Vertexer::initialiseVertexer(T&&... args)
{
  mTraits->initialise(std::forward<T>(args)...);
}

void Vertexer::dumpTraits()
{
  mTraits->dumpVertexerTraits();
}

std::vector<Vertex> Vertexer::exportVertices()
{
  std::vector<Vertex> vertices;
  for (auto& vertex : mTraits->getVertices()) {
    vertices.emplace_back(Point3D<float>(vertex.mX, vertex.mY, vertex.mZ), vertex.mRMS2, vertex.mContributors, vertex.mAvgDistance2);
    vertices.back().setTimeStamp(vertex.mTimeStamp);
  }
  return vertices;
}

template <typename... T>
float Vertexer::evaluateTask(void (Vertexer::*task)(T...), const char* taskName, std::ostream& ostream,
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

} // namespace ITS
} // namespace o2
#endif
