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

#include "TRKBase/Specs.h"

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace trk
{
enum eVDLayout {
  kIRIS4 = 0,
  kIRISFullCyl,
  kIRISFullCyl3InclinedWalls,
  kIRIS5,
  kIRIS4a,
};

enum eMLOTLayout {
  kCylindrical = 0,
  kSegmented,
  kSimplifiedRealistic,
};

enum eSrvLayout {
  kPeacockv1 = 0,
  kLOISymm,
};

struct TRKBaseParam : public o2::conf::ConfigurableParamHelper<TRKBaseParam> {
  std::string configFile = "";
  float serviceTubeX0 = 0.02f;                                          // X0 Al2O3
  float otBarrelWallThickness = 0.2f;                                   // cm, carbon fibre separation walls of the OT quarter barrels, 0 disables them
  float otEosCardCuThickness = constants::OT::eosCard::copperThickness; // cm, copper per plane in the OT end-of-stave card; drives the card x/X0
  bool irisOpen = false;
  bool includeLowServices = false;

  // Options for forward disks (FT3)
  int nTrapezoidalSegments = 32; // for the simple trapezoidal disks
  // Forward discs: define tolerance allowed for staves to go outside nominal radii
  double staveTolFT3MLInner = 0.;
  double staveTolFT3MLOuter = 0.;
  double staveTolFT3OTInner = 0.;
  double staveTolFT3OTOuter = 0.;

  // Forward discs: toggle to center staves at x=0 line
  bool placeSensorStackInMiddleOfStave = false;

  // Draw reference circles at inner and outer radius of forward discs for visualisation
  bool drawReferenceCircles = false;

  eVDLayout layoutVD = kIRIS4;         // VD detector layout design
  eMLOTLayout layoutMLOT = kSegmented; // ML and OT detector layout design
  eSrvLayout layoutSRV = kPeacockv1;   // Layout of services

  eVDLayout getLayoutVD() const { return layoutVD; }
  eMLOTLayout getLayoutMLOT() const { return layoutMLOT; }
  eSrvLayout getLayoutSRV() const { return layoutSRV; }

  O2ParamDef(TRKBaseParam, "TRKBase");
};

} // end namespace trk
} // end namespace o2

#endif