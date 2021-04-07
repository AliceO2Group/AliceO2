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

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{

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

  O2ParamDef(VertexerParamConfig, "ITSVertexerParam");
};

struct TrackerParamConfig : public o2::conf::ConfigurableParamHelper<TrackerParamConfig> {

  // Use TGeo for mat. budget
  bool useMatCorrTGeo = false;

  O2ParamDef(TrackerParamConfig, "ITSCATrackerParam");
};

} // namespace its
} // namespace o2
#endif