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
/// \file Configuration.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONFIGURATION_H_
#define TRACKINGITSU_INCLUDE_CONFIGURATION_H_

#include <array>
#include <climits>
#include <vector>
#include <cmath>

#include "ITStracking/Constants.h"

namespace o2
{
namespace its
{

template <typename Param>
class Configuration : public Param
{
 public:
  static Configuration<Param>& getInstance()
  {
    static Configuration<Param> instance;
    return instance;
  }
  Configuration(const Configuration<Param>&) = delete;
  const Configuration<Param>& operator=(const Configuration<Param>&) = delete;

 private:
  Configuration() = default;
};

struct TrackingParameters {
  TrackingParameters& operator=(const TrackingParameters& t);

  int CellMinimumLevel();

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  /// Trackleting cuts
  float TrackletMaxDeltaPhi = 0.3f;
  float TrackletMaxDeltaZ[constants::its::TrackletsPerRoad] = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f};
  /// Cell finding cuts
  float CellMaxDeltaTanLambda = 0.025f;
  float CellMaxDCA[constants::its::CellsPerRoad] = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f};
  float CellMaxDeltaPhi = 0.14f;
  float CellMaxDeltaZ[constants::its::CellsPerRoad] = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f};
  /// Neighbour finding cuts
  float NeighbourMaxDeltaCurvature[constants::its::CellsPerRoad - 1] = {0.008f, 0.0025f, 0.003f, 0.0035f};
  float NeighbourMaxDeltaN[constants::its::CellsPerRoad - 1] = {0.002f, 0.0090f, 0.002f, 0.005f};
};

struct MemoryParameters {
  /// Memory coefficients
  MemoryParameters& operator=(const MemoryParameters& t);
  int MemoryOffset = 256;
  float CellsMemoryCoefficients[constants::its::CellsPerRoad] = {2.3208e-08f, 2.104e-08f, 1.6432e-08f, 1.2412e-08f, 1.3543e-08f};
  float TrackletsMemoryCoefficients[constants::its::TrackletsPerRoad] = {0.0016353f, 0.0013627f, 0.000984f, 0.00078135f, 0.00057934f, 0.00052217f};
};

inline int TrackingParameters::CellMinimumLevel()
{
  return MinTrackLength - constants::its::ClustersPerCell + 1;
}

inline TrackingParameters& TrackingParameters::operator=(const TrackingParameters& t)
{
  this->ClusterSharing = t.ClusterSharing;
  this->MinTrackLength = t.MinTrackLength;
  /// Trackleting cuts
  this->TrackletMaxDeltaPhi = t.TrackletMaxDeltaPhi;
  for (int iT = 0; iT < constants::its::TrackletsPerRoad; ++iT)
    this->TrackletMaxDeltaZ[iT] = t.TrackletMaxDeltaZ[iT];
  /// Cell finding cuts
  this->CellMaxDeltaTanLambda = t.CellMaxDeltaTanLambda;
  this->CellMaxDeltaPhi = t.CellMaxDeltaPhi;
  for (int iC = 0; iC < constants::its::CellsPerRoad; ++iC) {
    this->CellMaxDCA[iC] = t.CellMaxDCA[iC];
    this->CellMaxDeltaZ[iC] = t.CellMaxDeltaZ[iC];
  }
  /// Neighbour finding cuts
  for (int iC = 0; iC < constants::its::CellsPerRoad - 1; ++iC) {
    this->NeighbourMaxDeltaCurvature[iC] = t.NeighbourMaxDeltaCurvature[iC];
    this->NeighbourMaxDeltaN[iC] = t.NeighbourMaxDeltaN[iC];
  }
  return *this;
}

inline MemoryParameters& MemoryParameters::operator=(const MemoryParameters& t)
{
  this->MemoryOffset = t.MemoryOffset;
  for (int iC = 0; iC < constants::its::CellsPerRoad; ++iC)
    this->CellsMemoryCoefficients[iC] = t.CellsMemoryCoefficients[iC];
  for (int iT = 0; iT < constants::its::TrackletsPerRoad; ++iT)
    this->TrackletsMemoryCoefficients[iT] = t.TrackletsMemoryCoefficients[iT];
  return *this;
}

struct VertexingParameters {
  float zCut = 0.002f;   //0.002f
  float phiCut = 0.005f; //0.005f
  float pairCut = 0.04f;
  float clusterCut = 0.8f;
  float histPairCut = 0.04f;
  float tanLambdaCut = 0.002f; // tanLambda = deltaZ/deltaR
  int clusterContributorsCut = 16;
  int phiSpan = -1;
  int zSpan = -1;
};

struct VertexerHistogramsConfiguration {
  VertexerHistogramsConfiguration() = default;
  VertexerHistogramsConfiguration(int nBins[3],
                                  int binSpan[3],
                                  float lowBoundaries[3],
                                  float highBoundaries[3]);
  int nBinsXYZ[3] = {402, 402, 4002};
  int binSpanXYZ[3] = {2, 2, 4};
  float lowHistBoundariesXYZ[3] = {-1.98f, -1.98f, -40.f};
  float highHistBoundariesXYZ[3] = {1.98f, 1.98f, 40.f};
  float binSizeHistX = (highHistBoundariesXYZ[0] - lowHistBoundariesXYZ[0]) / (nBinsXYZ[0] - 1);
  float binSizeHistY = (highHistBoundariesXYZ[1] - lowHistBoundariesXYZ[1]) / (nBinsXYZ[1] - 1);
  float binSizeHistZ = (highHistBoundariesXYZ[2] - lowHistBoundariesXYZ[2]) / (nBinsXYZ[2] - 1);
};

inline VertexerHistogramsConfiguration::VertexerHistogramsConfiguration(int nBins[3],
                                                                        int binSpan[3],
                                                                        float lowBoundaries[3],
                                                                        float highBoundaries[3])
{
  for (int i{0}; i < 3; ++i) {
    nBinsXYZ[i] = nBins[i];
    binSpanXYZ[i] = binSpan[i];
    lowHistBoundariesXYZ[i] = lowBoundaries[i];
    highHistBoundariesXYZ[i] = highBoundaries[i];
  }

  binSizeHistX = (highHistBoundariesXYZ[0] - lowHistBoundariesXYZ[0]) / (nBinsXYZ[0] - 1);
  binSizeHistY = (highHistBoundariesXYZ[1] - lowHistBoundariesXYZ[1]) / (nBinsXYZ[1] - 1);
  binSizeHistZ = (highHistBoundariesXYZ[2] - lowHistBoundariesXYZ[2]) / (nBinsXYZ[2] - 1);
}

struct VertexerStoreConfigurationGPU {
  VertexerStoreConfigurationGPU() = default;
  VertexerStoreConfigurationGPU(int cubBufferSize,
                                int maxTrkClu,
                                int cluLayCap,
                                int maxTrkCap,
                                int maxVert);

  // o2::its::GPU::Vector constructor requires signed size for initialisation
  int tmpCUBBufferSize = 25e5;
  int maxTrackletsPerCluster = 2e2;
  int clustersPerLayerCapacity = 4e4;
  int dupletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
  int processedTrackletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
  int maxTrackletCapacity = 2e4;
  int maxCentroidsXYCapacity = std::ceil(maxTrackletCapacity * (maxTrackletCapacity - 1) / 2);
  int nMaxVertices = 10;

  VertexerHistogramsConfiguration histConf;
};

inline VertexerStoreConfigurationGPU::VertexerStoreConfigurationGPU(int cubBufferSize,
                                                                    int maxTrkClu,
                                                                    int cluLayCap,
                                                                    int maxTrkCap,
                                                                    int maxVert) : tmpCUBBufferSize{cubBufferSize},
                                                                                   maxTrackletsPerCluster{maxTrkClu},
                                                                                   clustersPerLayerCapacity{cluLayCap},
                                                                                   maxTrackletCapacity{maxTrkCap},
                                                                                   nMaxVertices{maxVert}
{
  maxCentroidsXYCapacity = std::ceil(maxTrackletCapacity * (maxTrackletCapacity - 1) / 2);
  dupletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
  processedTrackletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CONFIGURATION_H_ */
