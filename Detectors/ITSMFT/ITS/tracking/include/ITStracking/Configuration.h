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
/// \file Configuration.h
/// \brief
///

#ifndef TRACKINGITSU_INCLUDE_CONFIGURATION_H_
#define TRACKINGITSU_INCLUDE_CONFIGURATION_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <array>
#include <climits>
#include <vector>
#include <cmath>
#endif

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
  TrackingParameters& operator=(const TrackingParameters& t) = default;

  int CellMinimumLevel();
  int CellsPerRoad() const { return NLayers - 2; }
  int TrackletsPerRoad() const { return NLayers - 1; }

  int NLayers = 7;
  int DeltaROF = 0;
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  std::vector<float> LayerxX0 = {5.e-3f, 5.e-3f, 5.e-3f, 1.e-2f, 1.e-2f, 1.e-2f, 1.e-2f};
  std::vector<float> LayerResolution = {5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f, 5.e-4f};
  std::vector<float> LayerMisalignment = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  int ZBins{256};
  int PhiBins{128};
  bool UseDiamond = false;
  float Diamond[3] = {0.f, 0.f, 0.f};

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  float NSigmaCut = 5;
  float PVres = 1.e-2f;
  /// Trackleting cuts
  float TrackletMinPt = 0.3f;
  /// Cell finding cuts
  float CellDeltaTanLambdaSigma = 0.007f;
  /// Fitter parameters
  bool UseMatBudLUT = false;
  unsigned long MaxMemory = 12000000000UL;
  std::array<float, 2> FitIterationMaxChi2 = {50, 20};
};

struct MemoryParameters {
  /// Memory coefficients
  MemoryParameters& operator=(const MemoryParameters& t) = default;
  int MemoryOffset = 256;
  std::vector<float> CellsMemoryCoefficients = {2.3208e-08f, 2.104e-08f, 1.6432e-08f, 1.2412e-08f, 1.3543e-08f};
  std::vector<float> TrackletsMemoryCoefficients = {0.0016353f, 0.0013627f, 0.000984f, 0.00078135f, 0.00057934f, 0.00052217f};
};

inline int TrackingParameters::CellMinimumLevel()
{
  return MinTrackLength - constants::its::ClustersPerCell + 1;
}

struct VertexingParameters {
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int ZBins{256};
  int PhiBins{128};

  float zCut = 0.002f;   // 0.002f
  float phiCut = 0.005f; // 0.005f
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

struct TimeFrameGPUConfig {
  TimeFrameGPUConfig() = default;
  TimeFrameGPUConfig(size_t cubBufferSize,
                     size_t maxTrkClu,
                     size_t cluLayCap,
                     size_t cluROfCap,
                     size_t maxTrkCap,
                     size_t maxVertCap);

  size_t tmpCUBBufferSize = 1e5; // In average in pp events there are required 4096 bytes
  size_t maxTrackletsPerCluster = 50;
  size_t clustersPerLayerCapacity = 5e5;
  size_t clustersPerROfCapacity = 1e4;
  size_t trackletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
  size_t maxLinesCapacity = 1e2;
  size_t maxCentroidsXYCapacity = std::ceil(maxLinesCapacity * (maxLinesCapacity - 1) / (float)2);
  size_t maxVerticesCapacity = 10;
  size_t nMaxROFs = 1e3;

  VertexerHistogramsConfiguration histConf; // <==== split into separate configs
};

inline TimeFrameGPUConfig::TimeFrameGPUConfig(size_t cubBufferSize,
                                              size_t maxTrkClu,
                                              size_t cluLayCap,
                                              size_t cluROfCap,
                                              size_t maxTrkCap,
                                              size_t maxVertCap) : tmpCUBBufferSize{cubBufferSize},
                                                                   maxTrackletsPerCluster{maxTrkClu},
                                                                   clustersPerLayerCapacity{cluLayCap},
                                                                   clustersPerROfCapacity{cluROfCap},
                                                                   maxLinesCapacity{maxTrkCap},
                                                                   maxVerticesCapacity{maxVertCap}
{
  maxCentroidsXYCapacity = std::ceil(maxLinesCapacity * (maxLinesCapacity - 1) / 2);
  trackletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
}

} // namespace its
} // namespace o2

#endif /* TRACKINGITSU_INCLUDE_CONFIGURATION_H_ */
