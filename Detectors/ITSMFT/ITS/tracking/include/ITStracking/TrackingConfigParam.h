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

#include <limits>
#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::its
{

struct VertexerParamConfig : public o2::conf::ConfigurableParamHelper<VertexerParamConfig> {
  bool saveTimeBenchmarks = false; // dump metrics on file

  int nIterations = 1;                     // Number of vertexing passes to perform.
  int vertPerRofThreshold = 0;             // Maximum number of vertices per ROF to trigger second a iteration.
  bool allowSingleContribClusters = false; // attempt to find vertices in case of a single tracklet found.
  int deltaRof = 0;                        // Number of ROFs to be considered for the vertexing.

  // geometrical cuts for tracklet selection
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
  int clusterContributorsCut = 16; // minimum number of contributors for the second vertex found in the same ROF (pileup cut)
  int maxTrackletsPerCluster = 1e2;
  int phiSpan = -1;
  int zSpan = -1;
  int ZBins = 1;     // z-phi index table configutation: number of z bins
  int PhiBins = 128; // z-phi index table configutation: number of phi bins

  bool useTruthSeeding{false};  // overwrite seeding vertices with MC truth
  bool outputContLabels{false}; // output additioanlly for each vertex its contributing line labels

  int nThreads = 1;
  bool printMemory = false;
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;

  O2ParamDef(VertexerParamConfig, "ITSVertexerParam");
};

struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig> {
  // Use TGeo for mat. budget
  static const int MaxIter = 4;
  static const int MinTrackLength = 4;
  static const int MaxTrackLength = 7;
  bool useMatCorrTGeo = false;                                              // use full geometry to corect for material budget accounting in the fits. Default is to use the material budget LUT.
  bool useFastMaterial = false;                                             // use faster material approximation for material budget accounting in the fits.
  int deltaRof = 0;                                                         // configure the width of the window in ROFs to be considered for the tracking.
  int minTrackLgtIter[MaxIter] = {};                                        // minimum track length at each iteration, used only if >0, otherwise use code defaults
  uint8_t startLayerMask[MaxIter] = {};                                     // mask of start layer for this iteration (if >0)
  float minPtIterLgt[MaxIter * (MaxTrackLength - MinTrackLength + 1)] = {}; // min.pT for given track length at this iteration, used only if >0, otherwise use code defaults
  float sysErrY2[7] = {0};                                                  // systematic error^2 in Y per layer
  float sysErrZ2[7] = {0};                                                  // systematic error^2 in Z per layer
  float maxChi2ClusterAttachment = -1.f;
  float maxChi2NDF = -1.f;
  float nSigmaCut = -1.f;
  float deltaTanLres = -1.f;
  float minPt = -1.f;
  float pvRes = -1.f;
  int LUTbinsPhi = -1;
  int LUTbinsZ = -1;
  float diamondPos[3] = {0.f, 0.f, 0.f}; // override the position of the vertex
  bool useDiamond = false;               // enable overriding the vertex position
  int useTrackFollower = -1;             // bit 0: allow mixing implies bits 1&2; bit 1: topwards; bit2: downwards; => 0 off
  float trackFollowerNSigmaZ = 1.f;      // sigma in z-cut for track-following search rectangle
  float trackFollowerNSigmaPhi = 1.f;    // sigma in phi-cut for track-following search rectangle
  float cellsPerClusterLimit = -1.f;
  float trackletsPerClusterLimit = -1.f;
  int findShortTracks = -1;
  int nROFsPerIterations = 0;              // size of the slice of ROFs to be processed at a time, preferably integer divisors of nROFs per TF, to balance the iterations.
  int nOrbitsPerIterations = 0;            // not implemented: size of the slice of ROFs to be processed at a time, computed using the number of ROFs per orbit.
  bool perPrimaryVertexProcessing = false; // perform the full tracking considering the vertex hypotheses one at the time.
  bool saveTimeBenchmarks = false;         // dump metrics on file
  bool overrideBeamEstimation = false;     // use beam position from meanVertex CCDB object
  int trackingMode = -1;                   // -1: unset, 0=sync, 1=async, 2=cosmics used by gpuwf only
  bool doUPCIteration = false;             // Perform an additional iteration for UPC events on tagged vertices. You want to combine this config with VertexerParamConfig.nIterations=2
  int nIterations = MaxIter;               // overwrite the number of iterations

  bool createArtefactLabels{false}; // create on-the-fly labels for the artefacts

  int nThreads = 1;
  bool printMemory = false;
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;
  bool fataliseUponFailure = true; // granular management of the fatalisation in async mode

  O2ParamDef(TrackerParamConfig, "ITSCATrackerParam");
};

struct ITSGpuTrackingParamConfig : public o2::conf::ConfigurableParamHelper<ITSGpuTrackingParamConfig> {
  static constexpr int MaxIter = TrackerParamConfig::MaxIter;

  /// Set nBlocks/nThreads to summarily override all kernel launch parameters in each iteration.
  /// Parameters must start with nBlocks/nThreads.
  static constexpr int OverrideValue{-1};
  static constexpr char const* BlocksName = "nBlocks";
  static constexpr char const* ThreadsName = "nThreads";
  int nBlocks = OverrideValue;
  int nThreads = OverrideValue;
  void maybeOverride() const;

  /// Individual kernel launch parameter for each iteration
  int nBlocksLayerTracklets[MaxIter] = {60, 60, 60, 60};
  int nThreadsLayerTracklets[MaxIter] = {256, 256, 256, 256};

  int nBlocksLayerCells[MaxIter] = {60, 60, 60, 60};
  int nThreadsLayerCells[MaxIter] = {256, 256, 256, 256};

  int nBlocksFindNeighbours[MaxIter] = {60, 60, 60, 60};
  int nThreadsFindNeighbours[MaxIter] = {256, 256, 256, 256};

  int nBlocksProcessNeighbours[MaxIter] = {60, 60, 60, 60};
  int nThreadsProcessNeighbours[MaxIter] = {256, 256, 256, 256};

  int nBlocksTracksSeeds[MaxIter] = {60, 60, 60, 60};
  int nThreadsTracksSeeds[MaxIter] = {256, 256, 256, 256};

  O2ParamDef(ITSGpuTrackingParamConfig, "ITSGpuTrackingParam");
};

} // namespace o2::its
#endif
