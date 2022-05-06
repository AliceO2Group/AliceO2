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

#ifndef O2_SIMCONFIG_SIMPARAM_H_
#define O2_SIMCONFIG_SIMPARAM_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2
{
namespace conf
{
// parameters to influence/set the tracking cuts in simulation
// (mostly used in O2MCApplication stepping)
struct SimCutParams : public o2::conf::ConfigurableParamHelper<SimCutParams> {
  bool stepFiltering = true; // if we activate the step filtering in O2BaseMCApplication
  bool trackSeed = false;    // per track seeding for track-reproducible mode

  double maxRTracking = 1E20;    // max R tracking cut in cm (in the VMC sense) -- applied in addition to cutting in the stepping function
  double maxAbsZTracking = 1E20; // max |Z| tracking cut in cm (in the VMC sense) -- applied in addition to cutting in the stepping function
  double ZmaxA = 1E20;           // max Z tracking cut on A side in cm -- applied in our custom stepping function
  double ZmaxC = 1E20;           // max Z tracking cut on C side in cm -- applied in out custom stepping function

  float maxRTrackingZDC = 50; // R-cut applied in the tunnel leading to ZDC when z > beampipeZ (custom stepping function)
  float tunnelZ = 1900;       // Z-value from where we apply maxRTrackingZDC (default value taken from standard "hall" dimensions)

  float globalDensityFactor = 1.f; // global factor that scales all material densities for systematic studies

  O2ParamDef(SimCutParams, "SimCutParams");
};

// parameter influencing material manager
struct SimMaterialParams : public o2::conf::ConfigurableParamHelper<SimMaterialParams> {
  // Local density value takes precedence over global density value, i.e. local values overwrite the global value.
  float globalDensityFactor = 1.f;
  std::string localDensityFactor; // Expected format: "SimMaterialParams.localDensityFactor=<mod1>:<value1>,<mod2>:<value2>,..."

  O2ParamDef(SimMaterialParams, "SimMaterialParams");
};

} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_SIMCUTPARAM_H_ */
