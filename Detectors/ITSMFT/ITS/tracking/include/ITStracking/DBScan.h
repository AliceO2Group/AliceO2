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
/// \file DBScan.h
/// \brief
///

#ifndef O2_ITS_TRACKING_DBSCAN_H_
#define O2_ITS_TRACKING_DBSCAN_H_

#include <algorithm>
#include "ITStracking/Graph.h"

namespace o2
{
namespace its
{

typedef std::pair<int, unsigned char> State;

template <typename T>
class DBScan : Graph<T>
{
 public:
  DBScan() = delete;
  explicit DBScan(const size_t nThreads);
  void init(std::vector<T>&, std::function<bool(const T& v1, const T& v2)>);
  void classifyVertices(const int);
  void classifyVertices(std::function<unsigned char(std::vector<Edge>&)> classFunction);
  void classifyVertices(std::function<unsigned char(std::vector<Edge>&)> classFunction, std::function<bool(State&, State&)> sortFunction);
  std::vector<State> getStates() const { return mStates; }
  std::vector<int> getCores();
  std::vector<std::vector<int>> computeClusters();

 private:
  std::vector<State> mStates;
  std::function<unsigned char(std::vector<Edge>&)> mClassFunction;
};

template <typename T>
DBScan<T>::DBScan(const size_t nThreads) : Graph<T>(nThreads)
{
}

template <typename T>
void DBScan<T>::init(std::vector<T>& vertices, std::function<bool(const T& v1, const T& v2)> discFunction)
{
  this->Graph<T>::init(vertices);
  this->Graph<T>::computeEdges(discFunction);
}

template <typename T>
void DBScan<T>::classifyVertices(const int nContributors)
{
  classifyVertices([nContributors](std::vector<o2::its::Edge>& edges) { return edges.size() == 0 ? 0 : edges.size() >= static_cast<size_t>(nContributors - 1) ? 2 : 1; },
                   [](State& s1, State& s2) { return static_cast<int>(s1.second) > static_cast<int>(s2.second); });
}

template <typename T>
void DBScan<T>::classifyVertices(std::function<unsigned char(std::vector<Edge>& edges)> classFunction)
{
  mClassFunction = classFunction;
  const size_t size = { this->mVertices->size() };
  mStates.resize(size);

  if (!this->isMultiThreading()) {
    for (size_t iVertex{ 0 }; iVertex < size; ++iVertex) {
      mStates[iVertex] = std::make_pair<int, unsigned char>(iVertex, classFunction(this->getEdges()[iVertex]));
    }
  } else {
    const size_t stride{ static_cast<size_t>(std::ceil(this->mVertices->size() / static_cast<size_t>(this->mExecutors.size()))) };
    for (size_t iExecutor{ 0 }; iExecutor < this->mExecutors.size(); ++iExecutor) {
      // We cannot pass a template function to std::thread(), using lambda instead
      this->mExecutors[iExecutor] = std::thread(
        [iExecutor, stride, this](const auto& classFunction) {
          for (size_t iVertex{ iExecutor * stride }; iVertex < stride * (iExecutor + 1) && iVertex < this->mVertices->size(); ++iVertex) {
            mStates[iVertex] = std::make_pair<int, unsigned char>(iVertex, classFunction(this->getEdges()[iVertex]));
          }
        },
        mClassFunction);
    }
  }
  for (auto&& thread : this->mExecutors) {
    thread.join();
  }
}

template <typename T>
void DBScan<T>::classifyVertices(std::function<unsigned char(std::vector<Edge>&)> classFunction, std::function<bool(State&, State&)> sortFunction)
{
  classifyVertices(classFunction);
  std::sort(mStates.begin(), mStates.end(), sortFunction);
}

template <typename T>
std::vector<int> DBScan<T>::getCores()
{
  std::vector<State> cores;
  std::vector<int> coreIndices;
  std::copy_if(mStates.begin(), mStates.end(), std::back_inserter(cores), [](const State& state) { return state.second == 2; });
  std::transform(cores.begin(), cores.end(), std::back_inserter(coreIndices), [](const State& state) -> int { return state.first; });
  return coreIndices;
}

template <typename T>
std::vector<std::vector<int>> DBScan<T>::computeClusters()
{
  std::vector<std::vector<int>> clusters;
  std::vector<int> cores = getCores();
  std::vector<unsigned char> usedVertices(this->mVertices->size(), false);

  for (size_t core{ 0 }; core < cores.size(); ++core) {
    if (!usedVertices[cores[core]]) {
      std::vector<unsigned char> clusterFlags = this->getCluster(cores[core]);
      std::transform(usedVertices.begin(), usedVertices.end(), clusterFlags.begin(), usedVertices.begin(), std::logical_or<>());
      clusters.emplace_back(std::move(this->getClusterIndices(clusterFlags)));
    }
  }
  return clusters;
}

struct Centroid final {
  Centroid() = default;
  Centroid(int* indices, float* position);
  void init();
  static float ComputeDistance(const Centroid& c1, const Centroid& c2);

  int mIndices[2];
  float mPosition[3];
};

} // namespace its
} // namespace o2
#endif