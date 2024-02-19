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
  int nRings = 21;
  int nTiles = 36;
  std::string configFile = "";

  O2ParamDef(RICHBaseParam, "RICHBase");
};

} // namespace rich
} // end namespace o2

#endif