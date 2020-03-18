// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSDPLTRACKINGPARAM_H_
#define ALICEO2_ITSDPLTRACKINGPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{

class VertexingParameters;
struct VertexerParamConfig : public o2::conf::ConfigurableParamHelper<VertexerParamConfig> {

  // geometrical cuts
  float zCut = 0.002f;   //0.002f
  float phiCut = 0.005f; //0.005f
  float pairCut = 0.04f;
  float clusterCut = 0.8f;
  float histPairCut = 0.04f;
  float tanLambdaCut = 0.002f; // tanLambda = deltaZ/deltaR
  int clusterContributorsCut = 16;
  int phiSpan = -1;
  int zSpan = -1;

  // histogram configuration
  int nBinsHistX = 402;
  int nBinsHistY = 402;
  int nBinsHistZ = 4002;
  int binSpanX = 2;
  int binSpanY = 2;
  int binSpanZ = 4;
  float lowHistBoundaryX = -1.98f;
  float lowHistBoundaryY = -1.98f;
  float lowHistBoundaryZ = -40.f;
  float highHistBoundaryX = 1.98f;
  float highHistBoundaryY = 1.98f;
  float highHistBoundaryZ = 40.f;

  // GPU configuration
  int gpuCUBBufferSize = 25e5;
  int gpuMaxTrackletsPerCluster = 2e2;
  int gpuMaxClustersPerLayer = 4e4;
  int gpuMaxTrackletCapacity = 2e4;
  int gpuMaxVertices = 10;

  O2ParamDef(VertexerParamConfig, "ITSVertexerParam");
};

// VertexerParamConfig VertexerParamConfig::sInstance;
} // namespace its
} // namespace o2
#endif