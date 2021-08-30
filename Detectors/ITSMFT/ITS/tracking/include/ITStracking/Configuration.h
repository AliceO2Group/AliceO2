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
  void CopyCuts(TrackingParameters& other, float scale = 1.)
  {
    TrackletMaxDeltaPhi = other.TrackletMaxDeltaPhi * scale;
    for (unsigned int ii{0}; ii < TrackletMaxDeltaZ.size(); ++ii) {
      TrackletMaxDeltaZ[ii] = other.TrackletMaxDeltaZ[ii] * scale;
    }
    CellMaxDeltaTanLambda = other.CellMaxDeltaTanLambda * scale;
    for (unsigned int ii{0}; ii < CellMaxDCA.size(); ++ii) {
      CellMaxDCA[ii] = other.CellMaxDCA[ii] * scale;
    }
    for (unsigned int ii{0}; ii < NeighbourMaxDeltaCurvature.size(); ++ii) {
      NeighbourMaxDeltaCurvature[ii] = other.NeighbourMaxDeltaCurvature[ii] * scale;
      NeighbourMaxDeltaN[ii] = other.NeighbourMaxDeltaN[ii] * scale;
    }
  }

  int CellMinimumLevel();
  int CellsPerRoad() const { return NLayers - 2; }
  int TrackletsPerRoad() const { return NLayers - 1; }

  int NLayers = 7;
  int DeltaROF = 1;
  std::vector<float> LayerZ = {16.333f + 1, 16.333f + 1, 16.333f + 1, 42.140f + 1, 42.140f + 1, 73.745f + 1, 73.745f + 1};
  std::vector<float> LayerRadii = {2.33959f, 3.14076f, 3.91924f, 19.6213f, 24.5597f, 34.388f, 39.3329f};
  int ZBins{256};
  int PhiBins{128};

  /// General parameters
  int ClusterSharing = 0;
  int MinTrackLength = 7;
  /// Trackleting cuts
  float TrackletMaxDeltaPhi = 0.3f;
  std::vector<float> TrackletMaxDeltaZ = {0.1f, 0.1f, 0.3f, 0.3f, 0.3f, 0.3f};
  /// Cell finding cuts
  float CellMaxDeltaTanLambda = 0.025f;
  std::vector<float> CellMaxDCA = {0.05f, 0.04f, 0.05f, 0.2f, 0.4f};
  float CellMaxDeltaPhi = 0.14f;
  std::vector<float> CellMaxDeltaZ = {0.2f, 0.4f, 0.5f, 0.6f, 3.0f};
  /// Neighbour finding cuts
  std::vector<float> NeighbourMaxDeltaCurvature = {0.008f, 0.0025f, 0.003f, 0.0035f};
  std::vector<float> NeighbourMaxDeltaN = {0.002f, 0.0090f, 0.002f, 0.005f};
  /// Fitter parameters
  bool UseMatBudLUT = false;
  std::array<float, 2> FitIterationMaxChi2 = {100, 50};
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

  // o2::its::gpu::Vector constructor requires signed size for initialisation
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
