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

#ifndef O2_TRK_BASEPARAM_H
#define O2_TRK_BASEPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trk
{

enum eOverallGeom {
  kDefaultRadii = 0, // After Upgrade Days March 2024
  kModRadii,
};

enum eLayout {
  kCylinder = 0,
  kTurboStaves,
  kStaggered,
};

struct TRKBaseParam : public o2::conf::ConfigurableParamHelper<TRKBaseParam> {
  std::string configFile = "";
  float serviceTubeX0 = 0.02f; // X0 Al2O3
  Bool_t irisOpen = false;

  eOverallGeom overallGeom = kDefaultRadii; // Overall geometry option, to be used in Detector::buildTRKMiddleOuterLayers

  eLayout layoutML = kTurboStaves; // Type of segmentation for the middle layers
  eLayout layoutOL = kStaggered;   // Type of segmentation for the outer layers

  eLayout getLayoutML() const { return layoutML; }
  eLayout getLayoutOL() const { return layoutOL; }

  O2ParamDef(TRKBaseParam, "TRKBase");
};

} // end namespace trk
} // end namespace o2

#endif