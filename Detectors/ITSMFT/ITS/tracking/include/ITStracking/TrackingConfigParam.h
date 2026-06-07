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
#include "ITStracking/Constants.h"

namespace o2::its
{

struct VertexerParamConfig : public o2::conf::ConfigurableParamHelper<VertexerParamConfig> {
  bool saveTimeBenchmarks = false; // dump metrics on file

  int nIterations = 1;         // Number of vertexing passes to perform.
  int vertPerRofThreshold = 0; // Maximum number of vertices per ROF to trigger second a iteration.

  // geometrical cuts for tracklet selection for Pb-Pb
  float zCut = 0.002f;
  float phiCut = 0.005f;
  float pairCut = 0.04f;
  float clusterCut = 0.170048f;
  float coarseZWindow = 0.055458f;
  float seedDedupZCut = 0.116685f;
  float refitDedupZCut = 0.039855f;
  float duplicateZCut = 0.200097f;
  float finalSelectionZCut = 0.034535f;
  float duplicateDistance2Cut = 0.005117f;
  float tanLambdaCut = 0.002f; // tanLambda = deltaZ/deltaR
  float nSigmaCut = 0.0164651f;
  float maxZPositionAllowed = 25.f; // 4x sZ of the beam

  // Artefacts selections
  int clusterContributorsCut = 2; // minimum number of contributors for an accepted final vertex
  int suppressLowMultDebris = 16; // suppress all vertices below this threshold if a vertex was already found in a rof
  int seedMemberRadiusTime = 0;
  int seedMemberRadiusZ = 2;
  int maxTrackletsPerCluster = 100;
  int phiSpan = -1;
  int zSpan = -1;
  int ZBins = 1;     // z-phi index table configutation: number of z bins
  int PhiBins = 128; // z-phi index table configutation: number of phi bins

  bool useTruthSeeding{false}; // overwrite seeding vertices with MC truth

  int nThreads = 1;
  bool printMemory = false;
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;

  O2ParamDef(VertexerParamConfig, "ITSVertexerParam");
};

struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig> {
  static const int MinTrackLength = 4;
  static const int MaxTrackLength = 7;

  bool useMatCorrTGeo = false;                                                         // use full geometry to corect for material budget accounting in the fits. Default is to use the material budget LUT.
  bool useFastMaterial = false;                                                        // use faster material approximation for material budget accounting in the fits.
  int addTimeError[7] = {0};                                                           // configure the width of the window in BC to be considered for the tracking.
  int minTrackLgtIter[constants::MaxIter] = {};                                        // minimum track length at each iteration, used only if >0, otherwise use code defaults
  uint8_t startLayerMask[constants::MaxIter] = {};                                     // mask of start layer for this iteration (if >0)
  int maxHolesIter[constants::MaxIter] = {};                                           // maximum number of missing internal layers allowed in the CA topology for each iteration
  uint16_t holeLayerMaskIter[constants::MaxIter] = {};                                 // layers that may be skipped by the CA topology for each iteration
  float minPtIterLgt[constants::MaxIter * (MaxTrackLength - MinTrackLength + 1)] = {}; // min.pT for given track length at this iteration, used only if >0, otherwise use code defaults
  float sysErrY2[7] = {0};                                                             // systematic error^2 in Y per layer
  float sysErrZ2[7] = {0};                                                             // systematic error^2 in Z per layer
  float maxChi2ClusterAttachment = -1.f;
  float maxChi2NDF = -1.f;
  float nSigmaCut = -1.f;
  float deltaTanLres = -1.f;
  float minPt = -1.f;
  float pvRes = -1.f;
  int LUTbinsPhi = -1;
  int LUTbinsZ = -1;
  float diamondPos[3] = {0.f, 0.f, 0.f};   // override the position of the vertex
  bool useDiamond = false;                 // enable overriding the vertex position
  bool perPrimaryVertexProcessing = false; // perform the full tracking considering the vertex hypotheses one at the time.
  bool saveTimeBenchmarks = false;         // dump metrics on file
  bool overrideBeamEstimation = false;     // use beam position from meanVertex CCDB object
  int trackingMode = -1;                   // -1: unset, 0=sync, 1=async, 2=cosmics used by gpuwf only
  bool doUPCIteration = false;             // Perform an additional iteration for UPC events on tagged vertices. You want to combine this config with VertexerParamConfig.nIterations=2
  int nIterations = constants::MaxIter;    // overwrite the number of iterations
  int reseedIfShorter = 6;                 // for the final refit reseed the track with circle if they are shorter than this value
  bool shiftRefToCluster{true};            // TrackFit: after update shift the linearization reference to cluster
  bool repeatRefitOut{false};              // repeat outward refit using inward refit as a seed
  bool createArtefactLabels{false};        // create on-the-fly labels for the artefacts

  int nThreads = 1;
  bool printMemory = false;
  size_t maxMemory = std::numeric_limits<size_t>::max();
  bool dropTFUponFailure = false;
  bool fataliseUponFailure = true;       // granular management of the fatalisation in async mode

  // Selections on tracks sharing clusters
  bool allowSharingFirstCluster = false; // allow first cluster sharing among tracks
  float sharedClusterMaxDeltaPhi = 0.05f; // Maximum allowed delta phi at the cluster position
  float sharedClusterMaxDeltaEta = 0.03f; // Maximum allowed delta eta at the cluster position
  bool sharedClusterOppositeSign = false; // Require opposite sign of the tracklets

  O2ParamDef(TrackerParamConfig, "ITSCATrackerParam");
};

} // namespace o2::its
#endif
