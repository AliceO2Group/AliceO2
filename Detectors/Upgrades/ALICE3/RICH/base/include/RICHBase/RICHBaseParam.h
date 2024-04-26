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

#ifndef O2_RICH_BASEPARAM_H
#define O2_RICH_BASEPARAM_H

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace rich
{
struct RICHBaseParam : public o2::conf::ConfigurableParamHelper<RICHBaseParam> {
  float zBaseSize = 18.4;        // cm
  float rMax = 117.0;            // cm
  float rMin = 90.0;             // cm
  float radiatorThickness = 2.0; // cm
  float detectorThickness = 0.2; // cm
  float zRichLength = 700.0;     // cm
  int nRings = 25;
  int nTiles = 36;
  bool oddGeom = false;

  // FWD and BWD RICH
  bool enableFWDRich = true;
  bool enableBWDRich = true;

  float rFWDMin = 13.7413f;
  float rFWDMax = 103.947f;

  // Aerogel:
  float zAerogelMin = 375.f;
  float zAerogelMax = 377.f;

  // Argon:
  float zArgonMin = 377.f;
  float zArgonMax = 407.f;

  // Detector:
  float zSiliconMin = 407.f;
  float zSiliconMax = 407.2f;

  O2ParamDef(RICHBaseParam, "RICHBase");
};

} // namespace rich
} // end namespace o2

#endif