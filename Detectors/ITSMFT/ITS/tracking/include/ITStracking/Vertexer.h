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
#include <array>
#include <iosfwd>

#include "ITStracking/ROframe.h"
#include "ITStracking/Constants.h"
#include "ITStracking/Configuration.h"
#include "ITStracking/VertexerTraits.h"
#include "ReconstructionDataFormats/Vertex.h"

// debug
#include "ITStracking/ClusterLines.h"
#include "ITStracking/Tracklet.h"
#include "ITStracking/Cluster.h"

namespace o2
{
namespace its
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

  float clustersToVertices(ROframe&, const bool useMc = false, std::ostream& = std::cout);

  template <typename... T>
  void initialiseVertexer(T&&... args);

  template <typename... T>
  void findTracklets(T&&... args);

  void findTrivialMCTracklets();

  void findVertices();

  // Utils
  void dumpTraits();
  template <typename... T>
  float evaluateTask(void (Vertexer::*)(T...), const char*, std::ostream& ostream, T&&... args);

  // debug, TBR
  std::vector<Line> getLines() const;
  std::vector<Tracklet> getTracklets01() const;
  std::vector<Tracklet> getTracklets12() const;
  std::array<std::vector<Cluster>, 3> getClusters() const;
  std::vector<std::array<float, 7>> getDeltaTanLambdas() const;
  std::vector<std::array<float, 4>> getCentroids() const;
  std::vector<std::array<float, 6>> getLinesData() const;
  void processLines();

 private:
  std::uint32_t mROframe = 0;
  VertexerTraits* mTraits = nullptr;
  VertexingParameters mVertParams;
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

inline void Vertexer::findTrivialMCTracklets()
{
  mTraits->computeTrackletsPureMontecarlo();
}

inline void Vertexer::dumpTraits()
{
  mTraits->dumpVertexerTraits();
}

inline std::vector<Vertex> Vertexer::exportVertices()
{
  std::vector<Vertex> vertices;
  for (auto& vertex : mTraits->getVertices()) {
    std::cout << "Emplacing vertex with: " << vertex.mContributors << " contribs" << std::endl;
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

// DEBUG
inline std::vector<Line> Vertexer::getLines() const
{
  return mTraits->mTracklets;
}

inline std::vector<Tracklet> Vertexer::getTracklets01() const
{
  return mTraits->mComb01;
}

inline std::vector<Tracklet> Vertexer::getTracklets12() const
{
  return mTraits->mComb12;
}

inline std::array<std::vector<Cluster>, 3> Vertexer::getClusters() const
{
  return mTraits->mClusters;
}

inline std::vector<std::array<float, 7>> Vertexer::getDeltaTanLambdas() const
{
  return mTraits->mDeltaTanlambdas;
}

inline std::vector<std::array<float, 4>> Vertexer::getCentroids() const
{
  return mTraits->mCentroids;
}

inline std::vector<std::array<float, 6>> Vertexer::getLinesData() const
{
  return mTraits->mLinesData;
}

inline void Vertexer::processLines()
{
  mTraits->processLines();
}

} // namespace its
} // namespace o2
#endif
