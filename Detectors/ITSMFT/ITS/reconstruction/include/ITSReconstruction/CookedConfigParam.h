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

#ifndef ALICEO2_COOKEDTRACKINGPARAM_H_
#define ALICEO2_COOKEDTRACKINGPARAM_H_

#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace its
{

struct CookedConfigParam : public o2::conf::ConfigurableParamHelper<CookedConfigParam> {
  // seed "windows" in z and phi: makeSeeds
  float zWin = 0.33;
  float minPt = 0.05;
  // Maximal accepted impact parameters for the seeds
  float maxDCAxy = 3.;
  float maxDCAz = 3.;
  // Space-point resolution
  float sigma = 0.0005;
  // Tracking "road" from layer to layer
  float roadY = 0.2;
  float roadZ = 0.3;
  // Minimal number of attached clusters
  int minNumberOfClusters = 4;

  O2ParamDef(CookedConfigParam, "ITSCookedTracker");
};

} // namespace its
} // namespace o2
#endif
