// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_SIMCONFIG_SIMCUTPARAM_H_
#define O2_SIMCONFIG_SIMCUTPARAM_H_

#include "SimConfig/ConfigurableParam.h"
#include "SimConfig/ConfigurableParamHelper.h"

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
  double ZmaxA = 1E20;           // max Z tracking cut on A side in cm -- applied in the stepping function
  double ZmaxC = 1E20;           // max Z tracking cut on C side in cm -- applied in the stepping function

  O2ParamDef(SimCutParams, "SimCutParams");
};
} // namespace conf
} // namespace o2

#endif /* O2_SIMCONFIG_SIMCUTPARAM_H_ */
