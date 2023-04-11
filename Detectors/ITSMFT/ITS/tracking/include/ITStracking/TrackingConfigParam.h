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

#ifndef ALICEO2_ITSDPLTRACKINGPARAM_H_
#define ALICEO2_ITSDPLTRACKINGPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{

struct VertexerParamConfig : public o2::conf::ConfigurableParamHelper<VertexerParamConfig> {

  bool allowSingleContribClusters = false;

  // geometrical cuts
  float zCut = 0.002f;
  float phiCut = 0.005f;
  float pairCut = 0.04f;
  float clusterCut = 0.8f;
  float histPairCut = 0.04f;
  float tanLambdaCut = 0.002f;      // tanLambda = deltaZ/deltaR
  float lowMultBeamDistCut = 0.1f;  // XY cut for low-multiplicity pile up
  int vertNsigmaCut = 4;            // N sigma cut for vertex XY
  float vertRadiusSigma = 0.05f;    // sigma of vertex XY
  float trackletSigma = 0.01f;      // tracklet to vertex sigma
  float maxZPositionAllowed = 25.f; // 4x sZ of the beam

  // Artefacts selections
  int clusterContributorsCut = 16;
  int maxTrackletsPerCluster = 1e2;
  int phiSpan = -1;
  int zSpan = -1;
  int ZBins = 1;
  int PhiBins = 128;

  int nThreads = 1;

  O2ParamDef(VertexerParamConfig, "ITSVertexerParam");
};

struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig> {

  // Use TGeo for mat. budget
  bool useMatCorrTGeo = false;
  bool useFastMaterial = false;
  float sysErrY2[7] = {0}; // systematic error^2 in Y per layer
  float sysErrZ2[7] = {0}; // systematic error^2 in Z per layer
  float maxChi2ClusterAttachment = -1.f;
  float maxChi2NDF = -1.f;
  float nSigmaCut = -1.f;
  float deltaTanLres = -1.f;
  float minPt = -1.f;
  float pvRes = -1.f;
  int LUTbinsPhi = -1;
  int LUTbinsZ = -1;
  float diamondPos[3] = {0.f, 0.f, 0.f};
  bool useDiamond = false;
  unsigned long maxMemory = 0;
  int useTrackFollower = -1;
  float cellsPerClusterLimit = -1.f;
  float trackletsPerClusterLimit = -1.f;
  int findShortTracks = -1;
  int nThreads = 1;

  O2ParamDef(TrackerParamConfig, "ITSCATrackerParam");
};

struct GpuRecoParamConfig : public o2::conf::ConfigurableParamHelper<GpuRecoParamConfig> {
  // GPU-specific parameters
  size_t tmpCUBBufferSize = 1e5; // In average in pp events there are required 4096 bytes
  size_t maxTrackletsPerCluster = 1e2;
  size_t clustersPerLayerCapacity = 2.5e5;
  size_t clustersPerROfCapacity = 1.5e3;
  size_t trackletsCapacity = maxTrackletsPerCluster * clustersPerLayerCapacity;
  size_t validatedTrackletsCapacity = 1e5;
  size_t cellsLUTsize = validatedTrackletsCapacity;
  size_t maxNeighboursSize = 1e4;
  size_t neighboursLUTsize = maxNeighboursSize;
  size_t maxLinesCapacity = 1e2;
  size_t maxVerticesCapacity = 5e4;
  size_t nTimeFramePartitions = 3;
  int maxGPUMemoryGB = -1;

  O2ParamDef(GpuRecoParamConfig, "ITSGpuRecoParam");
};

} // namespace its
} // namespace o2
#endif
